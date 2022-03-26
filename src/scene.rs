use std::{sync::Arc, mem};
use bytes::Bytes;
use openni2::Status;
use openni2::Stream;
use tokio::join;
use tokio::signal;
use tokio::time::Instant;
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
use image::Pixel;
use openni2::Device as NI2Device;
use openni2::SensorType;
use openni2::Frame;
use openni2::{OniRGB888Pixel, OniDepthPixel};
use crate::scene::tflite::Interpreter;

use std::convert::TryInto;
use std::borrow::Cow;
use core::array::IntoIter;

use std::io::Cursor;

use bytes::Buf;


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

struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>, //Bytes,
}

fn postprocess<'a>(results: Vec<Vec<f32>>) -> [u32; 224 * 224] {
    let dets = &results[4]; // first detection

    // I only am interested in the masks for now
    // Right now this is essentially semantic segmentation

    /*
    // ident4.chunks(81).map(|chunk| chunk[0])
    // identity4.chunks(81).map(|chunk| chunk[1]) = Red Robot
    // identity4.chunks(81).map(|chunk| chunk[2]) = Blue Robot
    // identity4.chunks(81).map(|chunk| chunk[3]) = Ball

    println!("{:?}", dets.chunks(81).map(|chunk| chunk.iter().map(|a| *a).take(4) // Debugging
    //.filter(|a| *a >= 0.0)
    .collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>().as_slice().chunks(28).collect::<Vec<&[Vec<f32>]>>());
    */

    let classes: [u8; 28 * 28] = dets.chunks(81).map(|chunk| { // TODO differentiate between instances
        let mut max = 0.0;
        let cls: [bool; 4] = chunk.iter().take(4).map(|a| {*a > max && {max = *a; true}})
                .collect::<Vec<bool>>().as_slice().try_into().unwrap();
        match cls {
            [false, true, false, false] => 1, // Red robot
            [false, _, true, false] => 2,
            [false, _, _, true] => 3,
            [false, _, _, _] | [_, _, _, _] => 0, // TODO first element
        }
    }).collect::<Vec<u8>>().as_slice().try_into().unwrap();

    /*for chunk in classes.as_slice().chunks(28).collect::<Vec<&[u8]>>().iter() {
        println!("{:?}", chunk);
    }*/ // Debugging model

    // TODO use confidence to shape mask
    let sample: [u32; 224 * 224] = classes.iter().map(|cls| [*cls as u32; 8]).flatten().collect::<Vec<u32>>().as_slice()
        .chunks(224).map(|chunk| [chunk; 8]).flatten().flatten().map(|r| *r).collect::<Vec<u32>>().try_into().unwrap();

    return sample;
}

fn classify_tile<'a>(frame_buffer: &mut [u32], interpreter: &mut Interpreter<'a, &'a BuiltinOpResolver>) {
    let data = {
        let mut i = -1;

        frame_buffer.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };
    // for now simply color
    let img = Image{
        width: 224, //640,
        height: 224, //480,
        channels: 3,
        data,
    };

    let tensor_index = interpreter.inputs()[0];
    let required_shape = interpreter.tensor_info(tensor_index).unwrap().dims;
    if img.height != required_shape[1]
            || img.width != required_shape[2]
            || img.channels != required_shape[3] {
        eprintln!("Input size mismatches:");
        eprintln!("\twidth: {} vs {}", img.width, required_shape[0]);
        eprintln!("\theight: {} vs {}", img.height, required_shape[1]);
        eprintln!("\tchannels: {} vs {}", img.channels, required_shape[2]);
    }

    let _start_time = Instant::now();
    interpreter.tensor_data_mut(tensor_index).unwrap()
        .copy_from_slice(img.data.as_ref());
    interpreter.invoke().expect("invoke failed");
    println!("eval time: {}μs", _start_time.elapsed().as_micros()); // ~50ms on edgetpu

    let outputs = interpreter.outputs();
    let mut results: Vec<Vec<f32>> = Vec::new();

    for &output in outputs {
       let tensor_info = interpreter.tensor_info(output).expect("must data");
       match tensor_info.element_kind {
           tflite::context::ElementKind::kTfLiteUInt8 => { // TODO
                let out_tensor: &[u8] = interpreter.tensor_data(output).expect("must data");
                let scale = tensor_info.params.scale;
                let zero_point = tensor_info.params.zero_point;
                results.push(out_tensor.into_iter()
                    .map(|&x| scale * (((x as i32) - zero_point) as f32)).collect());
            }
            tflite::context::ElementKind::kTfLiteFloat32 => {
                let out_tensor: &[f32] = interpreter.tensor_data(output).expect("must data");
                results.push(out_tensor.into_iter().copied().collect());
            }
            _ => eprintln!(
                "Tensor {} has unsupported output type {:?}.",
                tensor_info.name, tensor_info.element_kind,
            ),
        }
    }
    frame_buffer.copy_from_slice(&postprocess(results));
}

fn classify<'a>(frame_buffer: &mut [u32], interpreter: &mut Interpreter<'a, &'a BuiltinOpResolver>) {
    // println!("Classifying");

    let data = {
        let mut i = -1;

        frame_buffer.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };

    // // Debugging model
    // let image = image::DynamicImage::ImageRgb8(image::io::Reader::open("data/frc_balls.png").unwrap().decode().unwrap().to_rgb8());

    // Using a library to scale images.
    let image = image::DynamicImage::ImageRgb8(ImageBuffer::from_vec(640, 480, data).unwrap());
    let mut new_image = image.resize_exact(448, 224, image::imageops::FilterType::Triangle);

    // TODO: tile the image
    // 640 * 480 image
    // 224 * 224 sample
    let mut t1 = new_image.clone().crop(0, 0, 224, 224).as_bytes().chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();
    let mut t2 = new_image.crop(224, 0, 224, 224).as_bytes().chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();

    classify_tile(&mut t1, interpreter);
    classify_tile(&mut t2, interpreter);

    let img_buf = t1.chunks(224).collect::<Vec<&[u32]>>().iter().zip(t2.chunks(224).collect::<Vec<&[u32]>>().iter())
        .map(|(t1_horiz, t2_horiz)| [*t1_horiz, *t2_horiz].concat()).flatten(/*check op*/).collect::<Vec<u32>>();
    
    let data = {
        let mut i = -1;

        img_buf.iter().map( |px| {
            let out: [u8; 3] = u32::to_be_bytes(*px)[..3].iter().map(|byte| *byte) //as f32 / 255.0)
                .collect::<Vec<u8>>().as_slice().try_into().unwrap();
            out
        }).flatten().collect::<Vec<u8>>()
    };

    let image = image::DynamicImage::ImageRgb8(ImageBuffer::from_vec(448, 224, data).unwrap());
    let class_be = image.resize_exact(640, 480, image::imageops::FilterType::Triangle).as_bytes()
    .chunks(3).map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0])).collect::<Vec<u32>>();
    frame_buffer.copy_from_slice(&class_be);
}

/// Processes the camera input streams and detects targets.
pub(crate) async fn process_scene((point_cloud_queue, target_buffer_queue, not_empty): (&Arc<Mutex<Vec<[u16; 640 * 480]>>>, &Arc<Mutex<Vec<[u16; 640 * 480]>>>, Sender<()>)) {
    // This quick hack lets me implement Send for the variables that need to be held across the await.

    #[derive(Clone)]
    struct StreamSendDropper<'a>(Arc<Stream<'a>>);
    impl<'a> StreamSendDropper<'a>{ fn inner(self: Arc<Self>) -> Arc<Stream<'a>> { self.0.clone() } }
    impl<'a> Drop for StreamSendDropper<'a> { fn drop(&mut self) { self.0.stop(); } }
    #[derive(Clone)]
    struct StreamSendWrapper<'a>(Arc<StreamSendDropper<'a>>, Arc<NI2Device>);
    unsafe impl<'a> Send for StreamSendWrapper<'a> {}
    unsafe impl<'a> Sync for StreamSendWrapper<'a> {}
    impl<'a> StreamSendWrapper<'a>{ fn inner(self) -> Arc<Stream<'a>> { self.0.clone().inner().clone() } }

    #[derive(Clone)]
    struct DeviceSendWrapper(Arc<NI2Device>);
    unsafe impl Send for DeviceSendWrapper {}
    unsafe impl Sync for DeviceSendWrapper {}
    impl DeviceSendWrapper{
        fn inner(self) -> Arc<NI2Device> { self.0 }
        fn create_stream<'a>(&'a self, sensor_type: SensorType) -> StreamSendWrapper<'a> {
            StreamSendWrapper(Arc::new(StreamSendDropper(Arc::new(self.0.create_stream(sensor_type).unwrap()))), self.0.clone()) // for now unwrap
        }
    }

    // Load FlatBufferModel
    println!("{}", edgetpu::version());
    let model = FlatBufferModel::build_from_file(
       "data/FRC_model_edgetpu.tflite",
    ).expect("failed to load model");

    let resolver = BuiltinOpResolver::default();
    resolver.add_custom(edgetpu::custom_op(), edgetpu::register_custom_op());

    let builder = InterpreterBuilder::new(model, &resolver).expect("must create interpreter builder");

    let edgetpu_context = EdgeTpuContext::open_device().expect("failed to open coral device");

    let mut interpreter = builder.build().expect("must build interpreter");
    interpreter.set_external_context(
        tflite::ExternalContextType::EdgeTpu,
        edgetpu_context.to_external_context(),
    );
    interpreter.set_num_threads(4); // Max
    interpreter.allocate_tensors().expect("failed to allocate tensors.");

    // Load intel realsense (openni2 device)
    openni2::init().unwrap();
    let device = DeviceSendWrapper(Arc::new(NI2Device::open_default().unwrap()));
    let device = device.clone(); // for depth buffer ^^ for color buffer
    let depth = device.create_stream(SensorType::DEPTH);
    let color = device.create_stream(SensorType::COLOR); // for color buffer

    depth.clone().inner().start().expect("Failed to start depth stream");
    color.clone().inner().start().expect("Failed to start color stream");

    let mut instant = Instant::now();
    let mut frame = 0;
    loop {
        let mut buffer: [u32; 640 * 480] = unsafe { mem::zeroed() };

        { // Defining scope as to not hold non Send vars across await
            let color_frame = color.clone().inner().read_frame::<OniRGB888Pixel>().expect("Color frame not available to read.");

            //depth_histogram(&mut histogram, &depth_frame);
            for (i, color) in color_frame.pixels().iter().enumerate() {
                // Big endian ordering
                buffer[i] = ((color.r as u32) << 24) | ((color.g as u32) << 16) | ((color.b as u32) << 8);
            }
        }

        // Instance Segmentation
        // first 24 bits store true color and the last 8 store the class
        classify(&mut buffer, &mut interpreter); // Go through each element and make it into a u16
        let mut target_buffer: [u16; 640 * 480] = IntoIterator::into_iter(buffer).map(|px| ((px << 16) >> 16) as u16).collect::<Vec<u16>>().try_into().unwrap(); // buffer should be dropped here

        //drop(buffer);
        let depth_buffer: [u16; 640 * 480] = depth.clone().inner().read_frame::<OniDepthPixel>().expect("Depth frame not available to read.")
            .pixels().into_iter().map(|depth| *depth).collect::<Vec<u16>>().try_into().unwrap();

        // TODO: when no targets are found keep looking don't process scene.

        // append target and img and disparity
        let (mut target_buffer_queue_lock, mut point_cloud_queue_lock)
            = join!(target_buffer_queue.lock(), point_cloud_queue.lock());
        point_cloud_queue_lock.push(depth_buffer); // if I have time I can use color to help with optimization accuracy
        target_buffer_queue_lock.push(target_buffer); // all the points containing targets
        drop(point_cloud_queue_lock);
        drop(target_buffer_queue_lock);

        not_empty.try_send(()).unwrap_or(()); // Only will have an effect is append scene is stuck

        // Frame rate
        frame += 1;
        if frame % 60 == 0 {
            println!("fps: {}", 60.0/(instant.elapsed().as_micros() as f32 / 1000000.0));
            instant = Instant::now();
        }
    }
}

pub(crate) struct Scene {
    // pub(crate) map: Vec<(f32, f32, f32)>
    pub(crate) pos: Vec<(f32, f32)>,
    pub(crate) height: Vec<f32>,

    // Entities
    pub(crate) balls: Vec<(f32, f32)>,
    pub(crate) red_robots: Vec<(f32, f32)>,
    pub(crate) blue_robots: Vec<(f32, f32)>,

    // Speed
    pub(crate) connections: Vec<Vec<f32>>,

}

impl Scene {
   pub(crate) fn get_nearest_px(&self, pos: (f32, f32)) -> usize {
      todo!()
   }
   pub(crate) fn neighbors(&self, px: usize) -> Vec<usize> {
      todo!()
   }
}
/*
/// Builds on understanding of scene
/// Will Put PointCloud through a Point Cloud triangulation compute shader
pub(crate) async fn append_scene((point_cloud_queue, target_buffer_queue, not_empty): (Arc<Mutex<Vec<[u16; 640 * 480]>>>, Arc<Mutex<Vec<[u16; 640 * 480]>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>) {
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
            width: 640,
            height: 480,
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
        width: 640,
        height: 480,
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
        (0 .. 640 * 480 * 4).map(|_| 0u8)
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
        .dispatch([640 / 8, 480 / 8, 1]).unwrap()
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
    let scene_image = ImageBuffer::<Rgba<u8>, _>::from_raw(640, 480, &buffer_content[..]).unwrap();
    let filtered_scene_image = scene_image;

    #[inline]
    fn decode(px: [u8; 4]) -> (f32, f32, f32){
        // 4 bytes = 32 bits = 10.66 bits per dimension = 1024 values per dimension but if we agree to 1/z being stored we get a lot more accuracy from the points that matter
        (0.0, 0.0, 0.0)
    }
    // Pretransformed to position via giro input
    let new_scene_data = filtered_scene_image.pixels().map(|px| decode(px.0));

    let mut scene_lock = scene.lock().await;
    //scene_lock.from(new_scene_data);
}
*/
