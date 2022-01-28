use std::sync::Arc;
use eye_hal::PlatformContext;
use eye_hal::traits::{Context, Device as eyeDevice, Stream};
use image::imageops::FilterType;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::{time::Instant, sync::Mutex, net::TcpListener, io::{AsyncReadExt, AsyncWriteExt}, join};
use vulkano::{
    instance::{
        Instance,
        InstanceExtensions,
        PhysicalDevice
    },
    device::{
        Device,
        DeviceExtensions,
        Features
    },
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer
    },
    image::{
        StorageImage,
        Dimensions
    },
    format::{
        Format,
    },
    descriptor::{
        descriptor_set::PersistentDescriptorSet,
        PipelineLayoutAbstract
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        CommandBuffer
    },
    pipeline::ComputePipeline,
    sync::GpuFuture
};

use image::{
    ImageBuffer,
    Rgba, LumaA, GenericImageView
};


const MAP_PX_DROP: u32 = 8; // one pixel every 1024 with nearest neighbor in a ___ by ___ image


#[derive(PartialEq)]
enum TargetClass {
    Ball,
    Bot,
}

struct Target {
    class: TargetClass,
    pos: (f64, f64, f64)
}

struct TargetBuilder {
    class: TargetClass,
    pos: (f64, f64),
}

impl PartialEq for Target {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class && self.pos == other.pos
    }
}

struct Scene {
    map: Vec<(f32, f32, f32)> 
}
impl Scene{
    fn integrate(&mut self, new_scene_data: impl Iterator<Item = (f32, f32, f32)>){
        self.map = new_scene_data.collect(); // TODO devise plan
    }
}

struct Path {
    created: Instant,
    modified: Instant,
    directions: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_disparity_queue: Arc<Mutex<Vec<(Vec<u8>, u32)>>> = Arc::new(Mutex::new(Vec::new()));
    let (not_empty_tx, mut not_empty_rx) = mpsc::channel(1);
    let target_queue: Arc<Mutex<Vec<Target>>> = Arc::new(Mutex::new(Vec::new()));
    let scene: Arc<Mutex<Scene>> = Arc::new(Mutex::new(Scene{map: Vec::new()}));
    let path: Arc<Mutex<Path>> = {
        let created = Instant::now();
        Arc::new(Mutex::new(Path{
            created,
            modified: created,
            directions: Vec::new(),
        }))
    };

    tokio::spawn(handle_path_request(path.clone()));
    tokio::spawn(process_scene((img_disparity_queue.clone(), not_empty_tx), target_queue.clone()));
    loop {
        // figure out scene
        append_scene((img_disparity_queue.clone(), &mut not_empty_rx), scene.clone()).await;

        // build paths from what we know
        modify_path(path.clone(), target_queue.clone(), scene.clone()).await;
    }
}

async fn process_scene((img_disparity_queue, not_empty): (Arc<Mutex<Vec<(Vec<u8>, u32)>>>, Sender<()>), target_queue: Arc<Mutex<Vec<Target>>>) {
    // Turns out knowing depth will allow us to cheat a bit on this step. 
    // We will be able to use YOLO and not YOLACT for this stage. Know that
    // the efficiency of this approach is completely dependant
    async fn image_recognition (_frame_buffer: Vec<u8>) -> Vec<TargetBuilder> {
        // Because I have enough to do as is I will use an unoptimized YOLO library
        let im = read_bmp(manifest_dir().join("data/resized_cat.bmp")).expect("faild to load image");
        let model = FlatBufferModel::build_from_file(
            manifest_dir().join("data/mobilenet_v1_1.0_224_quant_edgetpu.tflite"),
        ).expect("failed to load model");
    
        let edgetpu_context = EdgeTpuContext::open_device().expect("failed to open coral device");
        let resolver = BuiltinOpResolver::default();
        resolver.add_custom(edgetpu::custom_op(), edgetpu::register_custom_op());
    
        let builder = InterpreterBuilder::new(model, &resolver).expect("must create interpreter builder");
        let mut interpreter = builder.build().expect("must build interpreter");
    
        interpreter.set_external_context(
            tflite::ExternalContextType::EdgeTpu,
            edgetpu_context.to_external_context(),
        );
        interpreter.set_num_threads(1);
        interpreter.allocate_tensors().expect("failed to allocate tensors.");
    
        let tensor_index = interpreter.inputs()[0];
        let required_shape = interpreter.tensor_info(tensor_index).unwrap().dims;
        if im.height != required_shape[1]
            || im.width != required_shape[2]
            || im.channels != required_shape[3]
        {
            eprintln!("Input size mismatches:");
            eprintln!("\twidth: {} vs {}", im.width, required_shape[0]);
            eprintln!("\theight: {} vs {}", im.height, required_shape[1]);
            eprintln!("\tchannels: {} vs {}", im.channels, required_shape[2]);
            return;
        }
    
        let inf_start = Instant::now();
        interpreter.tensor_data_mut(tensor_index).unwrap().copy_from_slice(im.data.as_ref());
        interpreter.invoke().expect("invoke failed");
        let outputs = interpreter.outputs();
        let mut results = Vec::new();
        for &output in outputs {
            let tensor_info = interpreter.tensor_info(output).expect("must data");
            match tensor_info.element_kind {
                tflite::context::ElementKind::kTfLiteUInt8 => {
                    let out_tensor: &[u8] = interpreter.tensor_data(output).expect("must data");
                    let scale = tensor_info.params.scale;
                    let zero_point = tensor_info.params.zero_point;
                    results = out_tensor.into_iter()
                        .map(|&x| scale * (((x as i32) - zero_point) as f32)).collect();
                }
                tflite::context::ElementKind::kTfLiteFloat32 => {
                    let out_tensor: &[f32] = interpreter.tensor_data(output).expect("must data");
                    results = out_tensor.into_iter().copied().collect();
                }
                _ => eprintln!(
                    "Tensor {} has unsupported output type {:?}.",
                    tensor_info.name, tensor_info.element_kind,
                ),
            }
        }
        let time_taken = inf_start.elapsed();
        let max = results
            .into_iter()
            .enumerate()
            .fold((0, -1.0), |acc, x| match x.1 > acc.1 {
                true => x,
                false => acc,
            });
        println!(
            "[Image analysis] max value index: {} value: {}",
            max.0, max.1
        );
        println!("Took {}ms", time_taken.as_millis());
    }

    fn find_pairs(_targets1: Vec<TargetBuilder>, _targets2: Vec<TargetBuilder>) -> Vec<(TargetBuilder, TargetBuilder)> {
        Vec::new()
    }

    fn calculate_disparity(targets1: Vec<TargetBuilder>, targets2: Vec<TargetBuilder>) -> (u32, Vec<Target>) {
        let _pair = find_pairs(targets1, targets2);
        (0, Vec::new())
    }

    let ctx = PlatformContext::default();

    // Query for available devices.
    let devices = ctx.devices().unwrap();

    let dev1 = ctx.open_device(&devices[0]).unwrap();
    let dev2 = ctx.open_device(&devices[1]).unwrap();

    // Query for available streams and just choose the first one.
    let streams_dev1 = dev1.streams().unwrap();
    let stream1_desc = streams_dev1[0].clone();
    let streams_dev2 = dev2.streams().unwrap();
    let stream2_desc = streams_dev2[0].clone();

    let mut stream1 = dev1.start_stream(&stream1_desc).unwrap();
    let mut stream2 = dev1.start_stream(&stream2_desc).unwrap();
    
    loop {
        let camera1_frame: Vec<u8> = stream1.next()
            .expect("Stream is dead")
            .expect("Failed to capture frame")
            .into_bytes().collect();
        let camera2_frame: Vec<u8> = stream2.next()
            .expect("Stream is dead")
            .expect("Failed to capture frame")
            .into_bytes().collect();
    
        // image recognition
        let (targets_2d1, targets_2d2) = join!(image_recognition(camera1_frame.clone()), image_recognition(camera2_frame));
        
        // disparity
        let (disparity, mut targets) = calculate_disparity(targets_2d1, targets_2d2);
    
        // append target and img and disparity
        let (mut target_queue_lock, mut img_disparity_queue_lock) = join!(target_queue.lock(), img_disparity_queue.lock());
        img_disparity_queue_lock.push((camera1_frame, disparity));
        target_queue_lock.append(&mut targets);

        not_empty.send(()).await.unwrap_or(());
        // target_queue_lock // remove duplicates
    }
}

async fn append_scene((img_disparity_queue, not_empty): (Arc<Mutex<Vec<(Vec<u8>, u32)>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>) {
    let instance = Instance::new(
        None,
        &InstanceExtensions::none(),
        None
    ).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_compute()).unwrap();
    
    let device_ext = DeviceExtensions{
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };

    let (device, mut queues) = Device::new(
        physical.clone(),
        &Features::none(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    let queue = queues.next().unwrap();

    let image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: 1024,
            height: 1024
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family())
    ).unwrap();

    mod cs {
        vulkano_shaders::shader!{
            ty: "compute",
            path: "./src/shader.glsl"
        }
    }

    let fractal_shader = cs::Shader::load(device.clone()).expect("failed to create fractal shader");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(
            device.clone(),
            &fractal_shader.main_entry_point(),
            &(),
            None,
        ).unwrap()
    );

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_image(image.clone()).unwrap()
            .build().unwrap()
    );

    let dest = CpuAccessibleBuffer::from_iter(
        device.clone(), 
        BufferUsage::all(), 
        false,
        (0 .. 1024 * 1024 * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::new(
        device.clone(),
        queue.family()
    ).unwrap();

    // wait for there to be something to pop
    if img_disparity_queue.lock().await.len() == 0 { not_empty.recv().await; }

    let (img, disparity) = img_disparity_queue.lock().await.pop().unwrap();
    builder
        .dispatch(
            [1024 / 8, 1024 / 8, 1],
            compute_pipeline.clone(), 
            set.clone(), 
            ()
        ).unwrap()
        .copy_image_to_buffer(
            image.clone(),
            dest.clone()
        ).unwrap();
    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = dest.read().unwrap();
    let scene_image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    let filtered_scene_image = image::imageops::resize(&scene_image, 1024/MAP_PX_DROP, 1024/MAP_PX_DROP, FilterType::Nearest);

    #[inline]
    fn decode(px: [u8; 4]) -> (f32, f32, f32){
        // 4 bytes = 32 bits = 10.66 bits per dimension = 1024 values per dimension but if we agree to 1/z being stored we get a lot more accuracy from the points that matter
        (0.0, 0.0, 0.0)
    }
    // Pretransformed to position via giro input
    let new_scene_data = filtered_scene_image.pixels().map(|px| decode(px.0));

    let mut scene_lock = scene.lock().await;
    scene_lock.integrate(new_scene_data);
}

async fn modify_path(path: Arc<Mutex<Path>>, target_queue: Arc<Mutex<Vec<Target>>>, scene: Arc<Mutex<Scene>>) {
    todo!();
}

async fn handle_path_request(path: Arc<Mutex<Path>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;

        let path = path.clone();
        tokio::spawn(async move {
            let mut buf = [0; 7];
            loop {
                match socket.read(&mut buf).await {
                    // socket closed
                    Ok(n) if n == 0 => return,
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("failed to read from socket; err = {:?}", e);
                        return;
                    }
                };
                let path = path.clone();
                match &buf {
                    b"NewPath" => {
                        let mut lock = path.lock().await;
                        *lock = {
                            let created = Instant::now();
                            Path {
                                created,
                                modified: created,
                                directions: Vec::new(),
                            }
                        };
                    },
                    b"GetPath" => {
                        let path_lock = path.lock().await;
                        if let Err(e) = socket.write_all(&path_lock.directions[..]).await {
                            eprintln!("failed to write to socket; err = {:?}", e);
                            return;
                        }
                    },
                    request => {
                        eprintln!("formatting err {:?} is not a request", std::str::from_utf8(request));
                        return;
                    }
                }
            }
        });
    }
}