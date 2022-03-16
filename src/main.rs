#[macro_use] extern crate lazy_static;

mod path;
pub(crate) mod scene;
pub(crate) mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/point_cloud_triangulation.comp"
    }
}

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::Instant;
use crate::path::Path;
use crate::scene::{Scene, Target};

use std::{thread, time};

const THREAD_STACK_SIZE: usize = 4 * 1024 * 1024; // 4Gb of mem

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point_cloud_queue: Arc<Mutex<Vec<[u16; 640 * 480]>>> = Arc::new(Mutex::new(Vec::new()));
    let target_buffer_queue: Arc<Mutex<Vec<[u16; 640 * 480]>>> = Arc::new(Mutex::new(Vec::new()));
    let (not_empty_tx, mut not_empty_rx) = mpsc::channel(1);
    let target_queue: Arc<Mutex<Vec<Target>>> = Arc::new(Mutex::new(Vec::new()));

    let scene: Arc<Mutex<Scene>> = Arc::new(Mutex::new(Scene{
       pos: Vec::new(), height: Vec::new(), 
       balls: Vec::new(), 
       red_robots: Vec::new(), blue_robots: Vec::new()
    }));
    let path: Arc<Mutex<Path>> = {
        let created = Instant::now();
        Arc::new(Mutex::new(Path{
            created,
            modified: created,
            directions: Vec::new(),
        }))
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(3)
        .thread_stack_size(THREAD_STACK_SIZE)
        .enable_io()
        .build().unwrap();
    
    rt.spawn(async move {
        path::handle_path_request(path.clone()).await
    });
    rt.spawn(async move {
        scene::process_scene(
            (
                &point_cloud_queue.clone(), 
                &target_buffer_queue.clone(), 
                not_empty_tx
            )
        ).await;
    });

    let heartbeat = time::Duration::from_millis(250);
    loop {
        thread::sleep(heartbeat);
        // figure out scene
        //scene::append_scene((point_cloud_queue.clone(), target_buffer_queue.clone(), &mut not_empty_rx), scene.clone()).await;

        // build paths from what we know
        // path::modify_path(path.clone(), target_queue.clone(), scene.clone()).await;
    }
}

