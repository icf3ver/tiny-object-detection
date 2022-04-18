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
use vulkano::sync;
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

use vulkano::descriptor_set::WriteDescriptorSet;

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

/// I don't have time to finish postprocessing so this will need to do for now
fn terrible_id (img: Vec<u8>) -> Vec<i8> {
    // Only run this on balls: For now it is the only item that needs unique ids
    let mut out = [-1; 28 * 28].to_vec();
    let mut id = -1;

    fn flood_fill(out: &mut Vec<i8>, start: usize, id: i8, img: &Vec<u8>) {
        let mut set = vec![start];
        
        let start_pos = 0;
        while let Some(px) = set.pop() {
            if img.get(px - 1) == Some(&3) {
                out[px - 1] = id;
                set.push(px - 1);
            }
            if img.get(px + 1) == Some(&3) {
                out[px + 1] = id;
                set.push(px + 1);
            }
            if img.get(px - 28) == Some(&3) {
                out[px - 28] = id;
                set.push(px - 28);
            }
            if img.get(px + 28) == Some(&3) {
                out[px + 28] = id;
                set.push(px + 28);
            }
        }
    }

    for (px, class) in img.iter().cloned().enumerate() {
        if class == 3 && out[px] == -1 {
            id += 1;
            flood_fill(&mut out, px, id, &img);
        }
    }
    
    out
}

fn postprocess<'a>(results: Vec<Vec<f32>>) -> [u32; 224 * 224] {
    let dets = &results[4]; // first detection

    // I only am interested in the prototype masks for now 
    // (Not enough time or resources to complete the yolact detection cleanup implementation)
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

    let mut ids = terrible_id(Vec::from(classes));

    /*for chunk in classes.as_slice().chunks(28).collect::<Vec<&[u8]>>().iter() {
        println!("{:?}", chunk);
    }*/ // Debugging model

    // TODO use confidence to shape mask // Note id: 0 is a ball id and the none id
    let sample: [u32; 224 * 224] = classes.iter().zip(&mut ids.into_iter()).map(|(cls, id)| [((*cls as u32) << 24 & (id as u32) << 16); 8]).flatten().collect::<Vec<u32>>().as_slice()
        .chunks(224).map(|chunk| [chunk; 8]).flatten().flatten().map(|r| *r).collect::<Vec<u32>>().try_into().unwrap();

    sample
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
    let image = image::DynamicImage::ImageRgb8(image::io::Reader::open("data/red_robot.png").unwrap().decode().unwrap().to_rgb8());

    // Using a library to scale images.
    //let image = image::DynamicImage::ImageRgb8(ImageBuffer::from_vec(640, 480, data).unwrap());
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
            .pixels().into_iter().cloned().collect::<Vec<u16>>().try_into().unwrap();

        
        let disp_img_buf = depth_buffer.iter().map(|i| (*i / 17) as u8).collect::<Vec<u8>>();
        let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
        image.save("depth.bmp").unwrap();


        // TODO: when no targets are found keep looking don't process scene.

        // append target and img and disparity
        let (mut target_buffer_queue_lock, mut point_cloud_queue_lock)
            = join!(target_buffer_queue.lock(), point_cloud_queue.lock());
        point_cloud_queue_lock.push(depth_buffer); // if I have time I can use color to help with optimization accuracy
        target_buffer_queue_lock.push(target_buffer); // all the points containing targets
        drop(point_cloud_queue_lock);
        drop(target_buffer_queue_lock);

        not_empty.try_send(()).unwrap_or(()); // Only will have an effect is append scene is stuck
	return;
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
    pub(crate) height: Vec<f32>, // index image
    pub(crate) pos: Vec<(f32, f32, f32)>, // The true positions of the pixels

    // Entities
    pub(crate) balls: Vec<(i32, i32)>,

    // Speed
    pub(crate) connections: Vec<[f32; 8]>,
}

impl Scene {
    pub(crate) fn neighbors(&self, px: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if px > 0 { out.push(px - 1); }
        if px < 680 * 480 - 1 { out.push(px + 1); }
        if px / 640 > 0 { out.push(px - 640); }
        if px / 640 < 480 - 1 { out.push(px + 640); }
        out
    }
}

/// Builds on understanding of scene
/// Will Put PointCloud through a Point Cloud triangulation compute shader
pub(crate) async fn append_scene((point_cloud_queue, target_buffer_queue, not_empty): (Arc<Mutex<Vec<[u16; 640 * 480]>>>, Arc<Mutex<Vec<[u16; 640 * 480]>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>) {
    let instance = Instance::new( // export DISPLAY=:0 over ssh
        None,
        Version::default(),
        &InstanceExtensions::none(),
        None
    ).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    // println!("{:?}", physical.supported_features());

    let queue_family = physical.queue_families()
        .find(|&q| q.supports_compute()).unwrap();

    let device_ext = DeviceExtensions{
        // khr_storage_buffer_storage_class: true,
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

    let dbg_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 640,
            height: 480,
            array_layers: 1,
        },
        Format::R32_UINT,
        Some(queue.family())
    ).unwrap();

    let map_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 640,
            height: 480,
            array_layers: 1,
        },
        Format::R32_UINT,
        Some(queue.family())
    ).unwrap();
    let world_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 640,
            height: 480,
            array_layers: 1,
        },
        Format::R32G32B32A32_SFLOAT,
        Some(queue.family())
    ).unwrap();
    let connections0_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 640,
            height: 480,
            array_layers: 1,
        },
        Format::R32G32B32A32_SFLOAT,
        Some(queue.family())
    ).unwrap();
    let connections1_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 640,
            height: 480,
            array_layers: 1,
        },
        Format::R32G32B32A32_SFLOAT,
        Some(queue.family())
    ).unwrap();

    let sampler_depth_img = Sampler::start(device.clone())
        .filter(Filter::Nearest)
        .address_mode(SamplerAddressMode::Repeat)
        .mip_lod_bias(1.0)
        .lod(0.0..=100.0)
        .build().unwrap();
    let sampler_class_img = Sampler::start(device.clone())
        .filter(Filter::Nearest)
        .address_mode(SamplerAddressMode::Repeat)
        .mip_lod_bias(1.0)
        .lod(0.0..=100.0)
        .build().unwrap();

    let point_cloud_triangulation_shader = crate::cs_triang::load(device.clone()).expect("failed to create shader");
    let parallel_weights_calculation_shader = crate::cs_weight::load(device.clone()).expect("failed to create shader");

    let compute_pipeline1 = ComputePipeline::new(
            device.clone(),
            point_cloud_triangulation_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).unwrap();
    let compute_pipeline2 = ComputePipeline::new(
            device.clone(),
            parallel_weights_calculation_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {}
        ).unwrap();
    
    let layout_p1 = compute_pipeline1.layout().descriptor_set_layouts().get(0).unwrap();
    let layout_p2 = compute_pipeline2.layout().descriptor_set_layouts().get(1).unwrap();

    // wait for there to be something to pop
    if point_cloud_queue.lock().await.len() == 0 { not_empty.recv().await; }

    let (mut point_cloud_queue_lock, mut target_buffer_queue_lock) = join!(point_cloud_queue.lock(), target_buffer_queue.lock());
    let color_depth_buffer = point_cloud_queue_lock.pop().unwrap().to_vec(); // TODO color
    let target_buffer = target_buffer_queue_lock.pop().unwrap().to_vec();
    // drop(point_cloud_queue_lock);
    // drop(target_buffer_queue_lock);

    let ball_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true, [[0.0,0.0,0.0,0.0]; 100]).unwrap();    

    let dimensions = ImageDimensions::Dim2d {
        width: 640,
        height: 480,
        array_layers: 1
    };

    let (depth_img, gpufuture_depth) = ImmutableImage::from_iter(
            color_depth_buffer.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R16_UINT,
            queue.clone(),
        ).unwrap();
    let (class_img, gpufuture_class) = ImmutableImage::from_iter(
            target_buffer.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8_UINT,
            queue.clone(),
        ).unwrap();

    let imgview_depth = ImageView::new(depth_img).unwrap();
    let imgview_class = ImageView::new(class_img).unwrap();

    let map_view1 = ImageView::new(map_image.clone()).unwrap();
    let dbg_view = ImageView::new(dbg_image.clone()).unwrap();
    
    let map_view2 = ImageView::new(map_image.clone()).unwrap();
    let world_view = ImageView::new(world_image.clone()).unwrap();
    let connections0_view = ImageView::new(connections0_image.clone()).unwrap();
    let connections1_view = ImageView::new(connections1_image.clone()).unwrap();

    let set1 = PersistentDescriptorSet::new(
        layout_p1.clone(),
        [
            WriteDescriptorSet::image_view_sampler(0, imgview_depth, sampler_depth_img),
            WriteDescriptorSet::image_view_sampler(1, imgview_class, sampler_class_img),

            WriteDescriptorSet::image_view(2, map_view1.clone()),
            // WriteDescriptorSet::image_view(3, connections0_view.clone()),
            // WriteDescriptorSet::image_view(4, connections1_view.clone()),

            WriteDescriptorSet::buffer(3, ball_buffer.clone()), // TODO buffer array
            WriteDescriptorSet::image_view(4, dbg_view.clone()),
        ]
    ).unwrap();
    let set2 = PersistentDescriptorSet::new(
        layout_p2.clone(),
        [
            WriteDescriptorSet::image_view(0, map_view2),
            WriteDescriptorSet::image_view(1, world_view),
            WriteDescriptorSet::image_view(2, connections0_view),
            WriteDescriptorSet::image_view(3, connections1_view),
        ]
    ).unwrap();

    let dbg_dest = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 640 * 480).map(|_| 0_u32) // or i32
    ).expect("failed to create buffer");

    let map_dest = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 640 * 480).map(|_| 0_u32) // or i32
    ).expect("failed to create buffer");
    let world_dest = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 640 * 480 * 4).map(|_| 0.0_f32)
    ).expect("failed to create buffer");
    let connections0_dest = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 640 * 480 * 4).map(|_| 0.0_f32)
    ).expect("failed to create buffer");
    let connections1_dest = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0 .. 640 * 480 * 4).map(|_| 0.0_f32)
    ).expect("failed to create buffer");

    let mut builder1 = AutoCommandBufferBuilder::primary(
        device.clone(), queue.family(), 
        CommandBufferUsage::MultipleSubmit,
    ).unwrap();
    builder1
        .bind_pipeline_compute(compute_pipeline1.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline1.layout().clone(),
            0,
            set1
        ).dispatch([640 / 8, 480 / 8, 1]).unwrap()
        .copy_image_to_buffer(
            map_image.clone(),
            map_dest.clone()
        ).unwrap()
        .copy_image_to_buffer(
            dbg_image.clone(),
            dbg_dest.clone()
        ).unwrap();
    let command_buffer1 = builder1.build().unwrap();

    // command_buffer.execute(queue.clone()).unwrap()
    //     .then_signal_fence_and_flush().unwrap()
    //     .wait(None).unwrap();

    let mut builder2 = AutoCommandBufferBuilder::primary(
        device.clone(), queue.family(), 
        CommandBufferUsage::MultipleSubmit,
    ).unwrap();
    builder2
        .bind_pipeline_compute(compute_pipeline2.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline2.layout().clone(),
            1,
            set2
        ).dispatch([640 / 8, 480 / 8, 1]).unwrap()
        .copy_image_to_buffer(
            world_image.clone(),
            world_dest.clone()
        ).unwrap()
        .copy_image_to_buffer(
            connections0_image.clone(),
            connections0_dest.clone()
        ).unwrap() // TODO use copy_image_to_buffer_dimensions
        .copy_image_to_buffer(
            connections1_image.clone(),
            connections1_dest.clone()
        ).unwrap();

    let command_buffer2 = builder2.build().unwrap();

    // command_buffer.execute(queue.clone()).unwrap()
    //     .then_signal_fence_and_flush().unwrap()
    //     .wait(None).unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer1).unwrap()
        .then_execute(queue.clone(), command_buffer2).unwrap()
        .then_signal_fence_and_flush().unwrap();
    
    future.wait(None).unwrap();
    
    println!("Completed Image Processing");
    
    let height_map_read = map_dest.read().unwrap();
    let world_read = world_dest.read().unwrap();
    let ball_buffer_read = ball_buffer.read().unwrap();

    let dbg_read = dbg_dest.read().unwrap();
    let disp_img_buf = dbg_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
    let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
    image.save("image.bmp").unwrap();


    let disp_img_buf = height_map_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
    let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
    image.save("map.bmp").unwrap();

    let connections = connections0_dest.read().unwrap()[..].chunks(4).into_iter()
        .zip(&mut connections1_dest.read().unwrap()[..].chunks(4).into_iter())
        .map(|(c1, c2)| [c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3]])
        .collect::<Vec<[f32; 8]>>();

    let mut scene_lock = scene.lock().await;
    *scene_lock = Scene {
        height: height_map_read[..].into_iter()
                .map(|i| *i as f32 * 480.0 / 2147483647.0 )
                .collect::<Vec<f32>>(),
        pos: world_read[..].chunks(4)
                .map(|chunk| (chunk[0], chunk[1], chunk[2]))
                .collect::<Vec<(f32, f32, f32)>>(),

        balls: ball_buffer_read[..].into_iter()
                .map(|chunk| (chunk[0] as i32, chunk[1] as i32))
                .collect::<Vec<(i32, i32)>>(),
        
        connections,
    };
}
