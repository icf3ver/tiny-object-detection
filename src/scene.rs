use std::{sync::Arc, mem};
use openni2::Stream;
use tokio::join;
use tokio::time::Instant;
use tokio::sync::{mpsc::{Receiver, Sender}, Mutex};
use vulkano::sync::GpuFuture;
use vulkano::device::Queue as VkoQueue;
use vulkano::pipeline::{Pipeline, PipelineBindPoint, ComputePipeline};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter};
use vulkano::image::{StorageImage, ImageDimensions, view::ImageView, ImmutableImage, MipmapsCount};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::sync;
use vulkano::format::Format;
use image::ImageBuffer;
use openni2::Device as NI2Device;
use openni2::SensorType;
use openni2::{OniRGB888Pixel, OniDepthPixel};
use crate::yolact::Yolact;

use vulkano::descriptor_set::WriteDescriptorSet;

use std::convert::TryInto;

#[allow(unused_imports)] // Debugging
use openni2::Status;
#[allow(unused_imports)] // Debugging
use std::io::Cursor;
#[allow(unused_imports)] // TOUSE
use core::array::IntoIter;
#[allow(unused_imports)] // Useful
use bytes::Buf;

/// Processes the camera input streams and detects targets.
#[allow(unused_assignments)] // TESTING
pub(crate) async fn process_scene((point_cloud_queue, target_buffer_queue, not_empty): (&Arc<Mutex<Vec<[u16; 640 * 480]>>>, &Arc<Mutex<Vec<[u16; 640 * 480]>>>, Sender<()>)) {
    // This quick hack lets me implement Send for the variables that need to be held across the await. and ensure

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
        #[allow(dead_code)] fn inner(self) -> Arc<NI2Device> { self.0 }
        fn create_stream<'a>(&'a self, sensor_type: SensorType) -> StreamSendWrapper<'a> {
            StreamSendWrapper(Arc::new(StreamSendDropper(Arc::new(self.0.create_stream(sensor_type).unwrap()))), self.0.clone()) // for now unwrap
        }
    }

    // Load FlatBufferModel
    println!("{}", edgetpu::version());
    let mut yolact = Yolact::init();

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
        yolact.classify(&mut buffer); // Go through each element and make it into a u16
        let target_buffer: [u16; 640 * 480] = IntoIterator::into_iter(buffer).map(|px| ((px << 16) >> 16) as u16).collect::<Vec<u16>>().try_into().unwrap(); // buffer should be dropped here

        //drop(buffer);
        let depth_buffer: [u16; 640 * 480] = depth.clone().inner().read_frame::<OniDepthPixel>().expect("Depth frame not available to read.")
            .pixels().into_iter().cloned().collect::<Vec<u16>>().try_into().unwrap();

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

        return; // TESTING
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
pub(crate) async fn append_scene((point_cloud_queue, target_buffer_queue, not_empty): (Arc<Mutex<Vec<[u16; 640 * 480]>>>, Arc<Mutex<Vec<[u16; 640 * 480]>>>, &mut Receiver<()>), scene: Arc<Mutex<Scene>>, vko_queue: Arc<VkoQueue>) {
    let vko_device = vko_queue.device().clone();

    let dims = ImageDimensions::Dim2d {width: 640, height: 480, array_layers: 1};

    let map_image = StorageImage::new(vko_device.clone(), dims, Format::R32_UINT, Some(vko_queue.family())).unwrap();
    let world_image = StorageImage::new(vko_device.clone(), dims, Format::R32G32B32A32_SFLOAT, Some(vko_queue.family())).unwrap();
    let connections0_image = StorageImage::new(vko_device.clone(), dims, Format::R32G32B32A32_SFLOAT, Some(vko_queue.family())).unwrap();
    let connections1_image = StorageImage::new(vko_device.clone(), dims, Format::R32G32B32A32_SFLOAT, Some(vko_queue.family())).unwrap();
    // // Heavy Debugging
    // let dbg_image = StorageImage::new(vko_device.clone(), dims, Format::R32_UINT, Some(vko_queue.family())).unwrap();

    let sampler_depth_img = Sampler::start(vko_device.clone()).filter(Filter::Nearest)
        .address_mode(SamplerAddressMode::Repeat)
        .mip_lod_bias(1.0).lod(0.0..=1000.0)
        .build().unwrap();
    let sampler_class_img = Sampler::start(vko_device.clone()).filter(Filter::Nearest)
        .address_mode(SamplerAddressMode::Repeat)
        .mip_lod_bias(1.0).lod(0.0..=1000.0)
        .build().unwrap();

    let cloud_shader = crate::cs_cloud::load(vko_device.clone()).expect("failed to create shader"); // TODO merge
    let cloud_weights_shader = crate::cs_cloud_weights::load(vko_device.clone()).expect("failed to create shader");
    // // Heavy Debugging
    // let dbg_shader = crate::cs_dbg::load(vko_device.clone()).expect("failed to create shader");

    let compute_pipeline1 = ComputePipeline::new(vko_device.clone(), cloud_shader.entry_point("main").unwrap(), &(), None, |_| {}).unwrap();
    let compute_pipeline2 = ComputePipeline::new(vko_device.clone(), cloud_weights_shader.entry_point("main").unwrap(), &(), None, |_| {}).unwrap();
    // // Heavy Debugging
    // let compute_pipeline3 = ComputePipeline::new(vko_device.clone(), dbg_shader.entry_point("main").unwrap(), &(), None, |_| {}).unwrap();

    let layout_p1 = compute_pipeline1.layout().descriptor_set_layouts().get(0).unwrap();
    let layout_p2 = compute_pipeline2.layout().descriptor_set_layouts().get(1).unwrap();
    // // Heavy Debugging
    // let layout_p3 = compute_pipeline3.layout().descriptor_set_layouts().get(2).unwrap();

    // Wait for there to be something to pop
    if point_cloud_queue.lock().await.len() == 0 { not_empty.recv().await; }
    let (mut point_cloud_queue_lock, mut target_buffer_queue_lock) = join!(point_cloud_queue.lock(), target_buffer_queue.lock());
    let depth_buffer = point_cloud_queue_lock.pop().unwrap().to_vec();
    let target_buffer = target_buffer_queue_lock.pop().unwrap().to_vec(); //.iter().map(|a| [(a >> 8) as u8, ((a << 8) >> 8) as u8]).collect::<Vec<[u8; 2]>>();
    // drop(point_cloud_queue_lock);
    // drop(target_buffer_queue_lock);

    // // Debugging
    // let disp_img_buf = depth_buffer.iter().map(|i| (*i / 17) as u8).collect::<Vec<u8>>();
    // let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
    // image.save("depth.bmp").unwrap();

    // Populate Samplers
    let (depth_img, gpufuture_depth) = ImmutableImage::from_iter(depth_buffer.iter().cloned(), dims, MipmapsCount::One, Format::R16_UINT, vko_queue.clone()).unwrap();
    let (class_img, gpufuture_class) = ImmutableImage::from_iter(target_buffer.iter().cloned(), dims, MipmapsCount::One, Format::R8G8_UINT, vko_queue.clone()).unwrap();
    let imgview_depth = ImageView::new(depth_img).unwrap();
    let imgview_class = ImageView::new(class_img).unwrap();

    // Create StorageImages
    let map_view = ImageView::new(map_image.clone()).unwrap();
    let world_view = ImageView::new(world_image.clone()).unwrap();
    let connections0_view = ImageView::new(connections0_image.clone()).unwrap();
    let connections1_view = ImageView::new(connections1_image.clone()).unwrap();
    // // Heavy Debugging
    // let dbg_view = ImageView::new(dbg_image.clone()).unwrap();

    // Create Buffers
    let ball_buffer = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), true, [[0.0,0.0,0.0,0.0]; 100]).unwrap();

    let set1 = PersistentDescriptorSet::new(layout_p1.clone(), [
        WriteDescriptorSet::image_view_sampler(0, imgview_depth.clone(), sampler_depth_img.clone()),
        WriteDescriptorSet::image_view_sampler(1, imgview_class.clone(), sampler_class_img.clone()),
        WriteDescriptorSet::image_view(2, map_view.clone()),
        WriteDescriptorSet::buffer(3, ball_buffer.clone()), // TODO buffer array
    ]).unwrap();
    let set2 = PersistentDescriptorSet::new(layout_p2.clone(), [
        WriteDescriptorSet::image_view(0, map_view.clone()),
        WriteDescriptorSet::image_view(1, world_view.clone()),
        WriteDescriptorSet::image_view(2, connections0_view.clone()),
        WriteDescriptorSet::image_view(3, connections1_view.clone()),
    ]).unwrap();
    // // Heavy Debugging
    // let set3 = PersistentDescriptorSet::new(layout_p3.clone(), [
    //     WriteDescriptorSet::image_view_sampler(0, imgview_depth.clone(), sampler_depth_img.clone()),
    //     WriteDescriptorSet::image_view(1, dbg_view.clone()),
    // ]).unwrap();

    let map_dest = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), false, (0 .. 640 * 480).map(|_| 0_u32)).expect("failed to create buffer");
    let world_dest = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), false, (0 .. 640 * 480 * 4).map(|_| 0.0_f32)).expect("failed to create buffer");
    let connections0_dest = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), false, (0 .. 640 * 480 * 4).map(|_| 0.0_f32)).expect("failed to create buffer");
    let connections1_dest = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), false, (0 .. 640 * 480 * 4).map(|_| 0.0_f32)).expect("failed to create buffer");
    // // Heavy Debugging
    // let dbg_dest = CpuAccessibleBuffer::from_iter(vko_device.clone(), BufferUsage::all(), false, (0 .. 640 * 480).map(|_| 0_u32)).expect("failed to create buffer");

    let mut builder1 = AutoCommandBufferBuilder::primary(vko_device.clone(), vko_queue.family(), CommandBufferUsage::MultipleSubmit).unwrap();
    builder1
        .bind_pipeline_compute(compute_pipeline1.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline1.layout().clone(),
            0, set1
        ).dispatch([640 / 8, 480 / 8, 1]).unwrap()
        .copy_image_to_buffer(map_image.clone(), map_dest.clone()).unwrap();
    let command_buffer1 = builder1.build().unwrap();

    let mut builder2 = AutoCommandBufferBuilder::primary(vko_device.clone(), vko_queue.family(), CommandBufferUsage::MultipleSubmit).unwrap();
    builder2
        .bind_pipeline_compute(compute_pipeline2.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline2.layout().clone(),
            1, set2
        ).dispatch([640 / 8, 480 / 8, 1]).unwrap()
        .copy_image_to_buffer(world_image.clone(), world_dest.clone()).unwrap()
        .copy_image_to_buffer(connections0_image.clone(), connections0_dest.clone()).unwrap() // TODO use copy_image_to_buffer_dimensions
        .copy_image_to_buffer(connections1_image.clone(), connections1_dest.clone()).unwrap();
    let command_buffer2 = builder2.build().unwrap();

    // // Heavy Debugging
    // let mut builder3 = AutoCommandBufferBuilder::primary(vko_device.clone(), vko_queue.family(), CommandBufferUsage::MultipleSubmit).unwrap();
    // builder3
    //     .bind_pipeline_compute(compute_pipeline3.clone())
    //     .bind_descriptor_sets(
    //         PipelineBindPoint::Compute,
    //         compute_pipeline3.layout().clone(),
    //         2, set3
    //     ).dispatch([640 / 8, 480 / 8, 1]).unwrap()
    //     .copy_image_to_buffer(dbg_image.clone(), dbg_dest.clone()).unwrap();
    // let command_buffer3 = builder3.build().unwrap();

    let future = sync::now(vko_device.clone())
        .join(gpufuture_depth) // Make sure to await these two futures
        .join(gpufuture_class)
        .then_execute(vko_queue.clone(), command_buffer1).unwrap()
        .then_execute(vko_queue.clone(), command_buffer2).unwrap()
        // // Heavy Debugging
        // .then_execute(vko_queue.clone(), command_buffer3).unwrap()
        .then_signal_fence_and_flush().unwrap();
    future.wait(None).unwrap();

    println!("Completed Image Processing");

    let height_map_read = map_dest.read().unwrap();
    let world_read = world_dest.read().unwrap();
    let ball_buffer_read = ball_buffer.read().unwrap();

    { // Debug Images
        let disp_img_buf = height_map_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
        let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
        image.save("map.bmp").unwrap();

        let connections0_read = connections0_dest.read().unwrap();
        let disp_img_buf = connections0_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
        let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
        image.save("connections0.bmp").unwrap();

        let connections1_read = connections1_dest.read().unwrap();
        let disp_img_buf = connections1_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
        let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
        image.save("connections1.bmp").unwrap();

        // // Heavy Debugging
        // let dbg_read = dbg_dest.read().unwrap();
        // let disp_img_buf = dbg_read.iter().map(|i| *i as u8).collect::<Vec<u8>>();
        // let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(640, 480, &disp_img_buf[..]).unwrap();
        // image.save("image.bmp").unwrap();
    }

    let height = height_map_read[..].into_iter()
        .map(|i| *i as f32)
        .collect::<Vec<f32>>();

    let pos = world_read[..].chunks(4)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect::<Vec<(f32, f32, f32)>>();

    let balls = ball_buffer_read[..].into_iter()
        .map(|chunk| (chunk[0] as i32, chunk[1] as i32))
        .collect::<Vec<(i32, i32)>>();

    let connections = connections0_dest.read().unwrap()[..].chunks(4).into_iter()
        .zip(&mut connections1_dest.read().unwrap()[..].chunks(4).into_iter())
        .map(|(c1, c2)| [c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3]])
        .collect::<Vec<[f32; 8]>>();

    let mut scene_lock = scene.lock().await;
    *scene_lock = Scene{height, pos, balls, connections};
}
