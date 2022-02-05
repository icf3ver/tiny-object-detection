use std::sync::Arc;
use edgetpu::tflite::FlatBufferModel;
use eye_hal::PlatformContext;
use eye_hal::traits::{Context, Device as eyeDevice, Stream};
use image::imageops::FilterType;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::{time::Instant, sync::Mutex, net::TcpListener, io::{AsyncReadExt, AsyncWriteExt}, join};
use vulkano::Version;
use vulkano::image::ImageViewAbstract;
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter};
use vulkano::{
    instance::{
        Instance,
        InstanceExtensions,
    },
    device::{
        physical::PhysicalDevice,
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
        ImageDimensions
    },
    format::{
        Format,
    },
    descriptor_set::{
        PersistentDescriptorSet
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        PrimaryCommandBuffer
    },
    pipeline::ComputePipeline,
    sync::GpuFuture
};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};

use image::{
    ImageBuffer,
    Rgba, LumaA, GenericImageView, ImageFormat
};


use bytes::Bytes;
use edgetpu::EdgeTpuContext;
use edgetpu::tflite;
use edgetpu::tflite::op_resolver::OpResolver;
use edgetpu::tflite::ops::builtin::BuiltinOpResolver;
use edgetpu::tflite::InterpreterBuilder;


const MAP_PX_DROP: u32 = 8; // one pixel every 1024 with nearest neighbor in a ___ by ___ image

// TODO calibrate
const Y_CAM: f32 = 0.0;
const BASELINE: f32 = 0.0;
const CALIBRATION_COEFFICIENT: u32 = 0;

#[derive(PartialEq)]
enum TargetClass {
    Ball,
    Bot,
}

struct Target {
    class: TargetClass,
    pos: (f32, f32, f32)
}

struct TargetBuilder {
    class: TargetClass,
    pos: (f32, f32),
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
    let img_disparity_queue: Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>> = Arc::new(Mutex::new(Vec::new()));
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

async fn process_scene((img_disparity_queue, not_empty): (Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>>, Sender<()>), target_queue: Arc<Mutex<Vec<Target>>>) {
    async fn image_recognition (_frame_buffer: Vec<u8>) -> Result<Vec<TargetBuilder>, ()> {
        let img;
        let model = FlatBufferModel::build_from_file(
            manifest_dir().join("data/todo.tflite"),
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
        if img.height != required_shape[1] 
                || img.width != required_shape[2] 
                || img.channels != required_shape[3] {
            eprintln!("Input size mismatches:");
            eprintln!("\twidth: {} vs {}", img.width, required_shape[0]);
            eprintln!("\theight: {} vs {}", img.height, required_shape[1]);
            eprintln!("\tchannels: {} vs {}", img.channels, required_shape[2]);
            return Err(());
        }

        // todo!();

        Ok(Vec::new())
    }

    fn find_pairs(_targets1: Vec<TargetBuilder>, _targets2: Vec<TargetBuilder>) -> Vec<(TargetBuilder, TargetBuilder)> {
        Vec::new()
    }

    fn calculate_disparity(targets1: Vec<TargetBuilder>, targets2: Vec<TargetBuilder>) -> Vec<Target> {
        let pair = find_pairs(targets1, targets2);
        let targets = Vec::new();
        for (tb_left, tb_right) in pair {
            let pos = (
                BASELINE * (tb_left.pos.0 + tb_right.pos.0) / (2.0 * (tb_left.pos.0 - tb_right.pos.0)), 
                BASELINE * Y_CAM / (tb_left.pos.0 - tb_right.pos.0), 
                BASELINE * CALIBRATION_COEFFICIENT as f32 / (tb_left.pos.0 - tb_right.pos.0)
            );
            targets.push(Target{
                class: tb_left.class,
                pos
            });
        }
        targets
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
        let (Ok(targets_2d1), Ok(targets_2d2)) = join!(
            image_recognition(camera1_frame.clone()), 
            image_recognition(camera2_frame.clone()));
        
        // disparity
        let mut targets = calculate_disparity(targets_2d1, targets_2d2);
    
        // append target and img and disparity
        let (mut target_queue_lock, mut img_disparity_queue_lock) = join!(target_queue.lock(), img_disparity_queue.lock());
        img_disparity_queue_lock.push((camera1_frame, camera2_frame));
        target_queue_lock.append(&mut targets);

        not_empty.send(()).await.unwrap_or(());
        // target_queue_lock // remove duplicates
    }
}

async fn append_scene((img_disparity_queue, not_empty): (Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>) {
    // I don't intend to recreate instance and structure every frame.
    // In a few commits when vk is functioning I will bring these to
    // main(). For now I want to keep this structure contained.
    
    // TODOs
    //  + Get new vko working

    let instance = Instance::new(
        None,
        Version::default(),
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
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: todo!(),
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family())
    ).unwrap();

    let sampler_img1 = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        vulkano::sampler::MipmapMode::Nearest,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        0.0,
        1.0,
        0.0,
        0.0
    ).unwrap();
    let sampler_img2 = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        vulkano::sampler::MipmapMode::Nearest,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        0.0,
        1.0,
        0.0,
        0.0
    ).unwrap();

    mod cs {
        vulkano_shaders::shader!{
            ty: "compute",
            path: "src/shader.glsl"
        }
    }

    let fractal_shader = cs::Shader::load(device.clone()).expect("failed to create fractal shader");

    let compute_pipeline = ComputePipeline::new(
            device.clone(),
            &fractal_shader.main_entry_point(),
            &(),
            None,
            |_| {}
        ).unwrap();

    let layout = compute_pipeline.layout().descriptor_set_layouts().get(0).unwrap();

    let (img_camera1_raw, img_camera2_raw) = img_disparity_queue.lock().await.pop().unwrap();

    let (image_camera1, gpufuture_camera1) = ImmutableImage::from_iter(
            img_camera1_raw.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8Srgb,
            queue.clone(),
        ).unwrap();
    let (image_camera2, gpufuture_camera2) = ImmutableImage::from_iter(
            img_camera2_raw.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8Srgb,
            queue.clone(),
        ).unwrap();

    let imgview_camera1 = ImageView::new(img_camera1).unwrap();
    let imgview_camera2 = ImageView::new(img_camera2).unwrap();

    let set = PersistentDescriptorSet::start(layout.clone())
        .add_sampled_image(imageview_camera1, sampler_img1).unwrap()
        .add_sampled_image(imageview_camera2, sampler_img2).unwrap()
        .add_image(image.clone()).unwrap()
        .build().unwrap();

    let dest = CpuAccessibleBuffer::from_iter(
        device.clone(), 
        BufferUsage::all(), 
        false,
        (0 .. 1024 * 1024 * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::MultipleSubmit,
    ).unwrap();

    // wait for there to be something to pop
    if img_disparity_queue.lock().await.len() == 0 { not_empty.recv().await; }

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set.clone()
        )
        .dispatch([1024 / 8, 1024 / 8, 1]).unwrap()
        .copy_image_to_buffer(
            image.clone(),
            dest.clone()
        ).unwrap();
    let command_buffer = builder.build().unwrap();

    command_buffer.execute(queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
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
                        socket.write(b"OK").await;
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
