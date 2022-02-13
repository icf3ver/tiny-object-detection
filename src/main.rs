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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point_cloud_queue: Arc<Mutex<Vec<[u32; 320 * 240]>>> = Arc::new(Mutex::new(Vec::new()));
    let target_buffer_queue: Arc<Mutex<Vec<[u32; 320 * 240]>>> = Arc::new(Mutex::new(Vec::new()));
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

    tokio::spawn(path::handle_path_request(path.clone()));
    tokio::spawn(scene::process_scene((point_cloud_queue.clone(), target_buffer_queue.clone(), not_empty_tx), target_queue.clone()));
    loop {
        // figure out scene
        scene::append_scene((point_cloud_queue.clone(), target_buffer_queue.clone(), &mut not_empty_rx), scene.clone()).await;

        // build paths from what we know
        path::modify_path(path.clone(), target_queue.clone(), scene.clone()).await;
    }
}

