use std::{sync::Arc, mem};
use bytes::Bytes;
use openni2::Status;
use openni2::Stream;
use tokio::join;
use tokio::signal;
use tokio::sync::{mpsc::{Receiver, Sender}, Mutex};
use edgetpu::EdgeTpuContext;
use edgetpu::tflite::{self, InterpreterBuilder, FlatBufferModel, op_resolver::OpResolver, ops::builtin::BuiltinOpResolver};
use vulkano::Version;
use vulkano::sync::GpuFuture;
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::device::{physical::PhysicalDevice, Device, DeviceExtensions, Features };
use vulkano::pipeline::{Pipeline, PipelineBindPoint, ComputePipeline};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter};
use vulkano::image::{StorageImage, ImageDimensions, ImageViewAbstract, view::ImageView, ImmutableImage, MipmapsCount};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::format::Format;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::{Rgba, GenericImageView};
use openni2::Device as NI2Device;
use openni2::SensorType;
use openni2::Frame;
use openni2::{OniRGB888Pixel, OniDepthPixel};

#[derive(PartialEq)]
enum TargetClass {
    Ball,
    RedRobot,
    BlueRobot,
}

pub(crate) struct Target {
    class: TargetClass,
    pos: (f32, f32, f32)
}

struct TargetBuilder {
    class: TargetClass,
    pos: (f32, f32),
}

/// Processes the camera input streams and detects targets.
pub(crate) async fn process_scene((point_cloud_queue, target_buffer_queue, not_empty): (Arc<Mutex<Vec<[u32; 320 * 240]>>>, Arc<Mutex<Vec<[u32; 320 * 240]>>>, Sender<()>), target_queue: Arc<Mutex<Vec<Target>>>) {
    struct Image {
        width: usize,
        height: usize,
        channels: usize,
        data: Bytes,
    }

    async fn classify(frame_buffer: &[u32; 320 * 240]) -> Result<([u32; 320 * 240], Vec<Target>), ()> {
        // for now simply color
        let img = Image{
            width: 320,
            height: 240,
            channels: 4,
            data: Bytes::from(frame_buffer.iter().map(|px| u32::to_be_bytes(*px)).flatten().collect::<Vec<u8>>())
        };
        let model = FlatBufferModel::build_from_file(
            "data/todo.tflite",
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

        // TODO: Working on dataset

        Ok(([0_u32; 320 * 240], Vec::new()))
    }

    pub fn depth_histogram(hist: &mut [f32], frame: &Frame<OniDepthPixel>) {
        let pixels = frame.pixels();
        let mut count = 0_usize;
        for h in hist.iter_mut() {
            *h = 0_f32;
        }

        for px in pixels {
            if *px != 0 {
                hist[*px as usize] += 1.0;
                count += 1;
            }
        }

        for i in 1..hist.len() {
            hist[i] += hist[i-1];
        }

        if count > 0 {
            for px in hist.iter_mut().skip(1) {
                *px = 256_f32 * (1.0_f32 - (*px / count as f32));
            }
        }
    }

    // This quick hack lets me implement Send for the variables that need to be held across the await.

    #[derive(Clone)]
    struct StreamSendWrapper<'a>(Arc<Stream<'a>>, Arc<NI2Device>);
    unsafe impl<'a> Send for StreamSendWrapper<'a> {}
    // unsafe impl<'a> Sync for StreamSendWrapper<'a> {}
    impl<'a> StreamSendWrapper<'a>{ fn inner(self) -> Arc<Stream<'a>> { self.0 } }

    #[derive(Clone)]
    struct DeviceSendWrapper(Arc<NI2Device>);
    unsafe impl Send for DeviceSendWrapper {}
    // unsafe impl Sync for DeviceSendWrapper {}
    impl DeviceSendWrapper{ 
        fn inner(self) -> Arc<NI2Device> { self.0 }
        fn create_stream<'a>(self, sensor_type: SensorType) -> StreamSendWrapper<'a> {
            todo!("patch lifetimes"); //StreamSendWrapper(Arc::new(self.0.create_stream(sensor_type).unwrap()), self.0.clone()) // for now unwrap
        }
    }

    openni2::init().unwrap();
    let device = DeviceSendWrapper(Arc::new(NI2Device::open_default().unwrap()));
    let depth = device.clone().create_stream(SensorType::DEPTH);
    let color = device.clone().create_stream(SensorType::COLOR);
    
    depth.clone().inner().start().expect("Failed to start depth stream");
    color.clone().inner().start().expect("Failed to start color stream");

    // async { // Should automatically be done
    //     signal::ctrl_c().await.expect("failed to listen for event");
    //     depth.stop();
    //     color.stop();
    //     eprintln!("Closed streams successfully");
    // };

    let mirror = color.clone().inner().get_mirroring().unwrap(); // for now unwrap
    let mut histogram: [f32; 10000] = unsafe { mem::zeroed() };

    // predicament: 32 bit system
    // images are formatted: 24 bit true color and 8 bit depth
    let mut buffer: [u32; 320 * 240] = unsafe { mem::zeroed() };
    loop {
        { // Defining scope as to not hold non Send vars across await
            let color_frame = color.clone().inner().read_frame::<OniRGB888Pixel>().expect("Color frame not available to read.");
            let depth_frame = depth.clone().inner().read_frame::<OniDepthPixel>().expect("Depth frame not available to read.");
            depth_histogram(&mut histogram, &depth_frame);
            for (i, (color, depth)) in color_frame.pixels().iter().zip(depth_frame.pixels()).enumerate() {
                if *depth > 0 {
                    // let brightness = (depth / 256) as u32;
                    let brightness = histogram[*depth as usize] as u32;
                    // Big endian ordering
                    buffer[i] = ((color.r as u32) << 24) | ((color.g as u32) << 16) | ((color.b as u32) << 8) | (brightness as u32);
                } else {
                    buffer[i] = 0;
                }
            }
        }

        // Instance Segmentation
        // first 24 bits store center of object and the last 8 store the class
        let (target_buffer, mut target_vec): ([u32; 320 * 240], Vec<Target>) = classify(&mut buffer).await.unwrap(); // unwrap for now 
    
        // append target and img and disparity
        let (mut target_queue_lock, mut target_buffer_queue_lock, mut point_cloud_queue_lock) 
            = join!(target_queue.lock(), target_buffer_queue.lock(), point_cloud_queue.lock());
        point_cloud_queue_lock.push(buffer);
        target_buffer_queue_lock.push(target_buffer);
        target_queue_lock.append(&mut target_vec);
        drop(point_cloud_queue_lock);
        drop(target_buffer_queue_lock);
        drop(target_queue_lock);

        not_empty.try_send(()).unwrap_or(()); // Only will have an effect is append scene is stuck
    }
}

pub(crate) struct Scene {
    pub(crate) map: Vec<(f32, f32, f32)> 
}
impl Scene{
    fn integrate(&mut self, new_scene_data: impl Iterator<Item = (f32, f32, f32)>){
        self.map = new_scene_data.collect(); // TODO devise plan
    }
}

/// Builds on understanding of scene
/// Will Put PointCloud through a Point Cloud triangulation compute shader
pub(crate) async fn append_scene((point_cloud_queue, target_buffer_queue, not_empty): (Arc<Mutex<Vec<[u32; 320 * 240]>>>, Arc<Mutex<Vec<[u32; 320 * 240]>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>) {
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

    // TODO: Remnants of last shader

    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 320,
            height: 240,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
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

    let fractal_shader = crate::cs::load(device.clone()).expect("failed to create shader");

    let compute_pipeline = ComputePipeline::new(
            device.clone(),
            fractal_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).unwrap();

    let layout = compute_pipeline.layout().descriptor_set_layouts().get(0).unwrap();

    let (mut point_cloud_queue_lock, mut target_buffer_queue_lock) = join!(point_cloud_queue.lock(), target_buffer_queue.lock());
    let color_depth_buffer = point_cloud_queue_lock.pop().unwrap();
    let target_buffer = target_buffer_queue_lock.pop().unwrap();
    drop(point_cloud_queue_lock);
    drop(target_buffer_queue_lock);
    
    let dimensions = ImageDimensions::Dim2d {
        width: 320,
        height: 240,
        array_layers: 1
    };
    let (color_depth_img, gpufuture_color_depth) = ImmutableImage::from_iter(
            color_depth_buffer.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        ).unwrap();
    let (target_img, gpufuture_target) = ImmutableImage::from_iter(
            target_buffer.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        ).unwrap();

    let imgview_color_depth = ImageView::new(color_depth_img).unwrap();
    let imgview_target = ImageView::new(target_img).unwrap();

    let mut set = PersistentDescriptorSet::start(layout.clone());
    let set = set
        .add_sampled_image(imgview_color_depth, sampler_img1).unwrap()
        .add_sampled_image(imgview_target, sampler_img2).unwrap()
        .add_image(ImageView::new(image.clone()).unwrap()).unwrap();
    let set = todo!("build"); //(&set).build();

    let dest = CpuAccessibleBuffer::from_iter(
        device.clone(), 
        BufferUsage::all(), 
        false,
        (0 .. 320 * 240 * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::MultipleSubmit,
    ).unwrap();

    // wait for there to be something to pop
    if point_cloud_queue.lock().await.len() == 0 { not_empty.recv().await; }

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            todo!() //set//.clone()
        )
        .dispatch([320 / 8, 240 / 8, 1]).unwrap()
        .copy_image_to_buffer(
            image.clone(),
            dest.clone()
        ).unwrap();
    let command_buffer = builder.build().unwrap();

    command_buffer.execute(queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // let buffer_content = dest.read().unwrap();
    // let scene_image = ImageBuffer::<Rgba<u8>, _>::from_raw(320, 240, &buffer_content[..]).unwrap();
    
    let buffer_content = dest.read().unwrap();
    let scene_image = ImageBuffer::<Rgba<u8>, _>::from_raw(320, 240, &buffer_content[..]).unwrap();
    let filtered_scene_image = scene_image;

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
