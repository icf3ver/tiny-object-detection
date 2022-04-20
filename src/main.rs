mod path;
pub mod yolact; // Unfinished
pub(crate) mod scene;
pub(crate) mod cs_cloud {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "shaders/pt_cloud.comp"
    }
}
pub(crate) mod cs_cloud_weights {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "shaders/pt_cloud_weights.comp"
    }
}
pub(crate) mod cs_dbg {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "shaders/dbg.comp"
    }
}

use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{mpsc, Mutex};
use crate::path::Path;
use crate::scene::Scene;

use vulkano::instance::{Instance as VkoInstance, InstanceExtensions};
use vulkano::Version as VkVersion;
use vulkano::device::{DeviceExtensions, Features, Device as VkoDevice, Queue as VkoQueue, physical::PhysicalDevice};

#[tokio::main]
#[allow(unreachable_code)] // TESTING
async fn manage(vko_queue: Arc<VkoQueue>, tokio_rt: tokio::runtime::Runtime) -> Result<(), Box<dyn std::error::Error>> {
    let point_cloud_queue: Arc<Mutex<Vec<[u16; 640 * 480]>>> = Arc::new(Mutex::new(Vec::new()));
    let target_buffer_queue: Arc<Mutex<Vec<[u16; 640 * 480]>>> = Arc::new(Mutex::new(Vec::new()));
    let (not_empty_tx, mut not_empty_rx) = mpsc::channel(1);

    let scene: Arc<Mutex<Scene>> = Arc::new(Mutex::new(Scene{
       pos: Vec::new(), height: Vec::new(), 
       balls: Vec::new(),
       
       connections: Vec::new()
    }));

    let path: Arc<Mutex<Path>> = {
        let created = SystemTime::now();
        Arc::new(Mutex::new(Path{
            created,
            // modified: created,
            directions: Vec::new(),
        }))
    };
    
    tokio_rt.spawn({
        let path_clone = path.clone();
        async move {
            path::handle_path_request(path_clone).await
        }
    });

    tokio_rt.spawn({
        let target_buffer_queue_clone = target_buffer_queue.clone();
        let point_cloud_queue_clone = point_cloud_queue.clone();
        async move {
            scene::process_scene(
                (
                    &point_cloud_queue_clone, 
                    &target_buffer_queue_clone,
                    not_empty_tx
                )
            ).await;
        }
    });

    //let heartbeat = time::Duration::from_millis(250);
    loop {
        //thread::sleep(heartbeat);

        // figure out scene
        scene::append_scene(
            (
                point_cloud_queue.clone(), 
                target_buffer_queue.clone(), 
                &mut not_empty_rx
            ), 
            scene.clone(),
            vko_queue.clone()
        ).await;
        
        panic!("Completed Append Scene.");

        // build paths from what we know
        path::modify_path(path.clone(), scene.clone()).await;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vulkano Interfacing
    let vko_instance = VkoInstance::new( 
        // Make sure to export DISPLAY=:0 if you are working over ssh
        // and configure `/boot/config.txt` if no screen is present
        None,
        // VkVersion{ major: 1, minor: 1, patch: 97 },
        VkVersion::default(), 
        &InstanceExtensions::none(),
        None
    ).expect("failed to create instance");
    
    let (_vko_device, mut vko_queues) = {
        let physical = PhysicalDevice::enumerate(&vko_instance).next().unwrap();

        let queue_family = physical.queue_families()
            .find(|&q| q.supports_compute()).unwrap();

        VkoDevice::new(
            physical.clone(),
            &Features::none(),
            &DeviceExtensions{
                // khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned()
        ).unwrap()
    };
    let vko_queue = vko_queues.next().unwrap();

    // Tokio Runtime
    const THREAD_STACK_SIZE: usize = 4 * 1024 * 1024; // 4Gb
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(3)
        .thread_stack_size(THREAD_STACK_SIZE)
        .enable_io()
        .build().unwrap();
    
    // Begin
    manage(vko_queue, tokio_rt)
}