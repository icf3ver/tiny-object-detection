use std::io::Cursor;
use std::sync::Arc;
use edgetpu::tflite::FlatBufferModel;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::image::{ImmutableImage, MipmapsCount};
use eye_hal::PlatformContext;
use vulkano::image::view::ImageView;
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

use image::io::Reader as ImageReader;

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

struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Bytes,
}

async fn process_scene((img_disparity_queue, not_empty): (Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>>, Sender<()>), target_queue: Arc<Mutex<Vec<Target>>>) {
    async fn image_recognition (frame_buffer: Vec<u8>) -> Result<Vec<TargetBuilder>, ()> {
        let img = Image{
            width: 1024,
            height: 1024,
            channels: 3,
            data: Bytes::from(frame_buffer)
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
    // In a few commits when