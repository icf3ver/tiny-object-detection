use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::time::Instant;
use crate::scene::{Scene, Target};
use std::collections::BTreeSet;

pub(crate) struct Path {
    pub(crate) created: Instant,
    // pub(crate) modified: Instant,
    pub(crate) directions: Vec<(f32, f32)>,
}
impl Path {
    fn serialize(&self) -> [u8; 12] { 
        todo!()
    }
}

/// Modifies the Path with the new scene information
pub(crate) async fn modify_path(arc_path: Arc<Mutex<Path>>, target_queue: Arc<Mutex<Vec<Target>>>, scene: Arc<Mutex<Scene>>) {
    let mut scene_lock = scene.lock().await;
    
    let mut set = Vec::new(); // TODO use TreeSet (first & last requires nightly)
    let mut path: [usize; 224 * 224] = [usize::MAX - 1; 224 * 224]; // MAX-1 = undefined MAX-2 = target (I do not need u32 size)
    let mut cost: [f32; 224 * 224] = [f32::MAX; 224 * 224]; // Cost to travel from a point

    let mut ball_checked: [u8; 224 * 112] = [0; 224 * 112];
    let mut ball: [u8; 224 * 56] = [0; 224 * 56];

    let ball_poss = &scene_lock.balls[..3];
    let dest_nodes = ball_poss.iter().map(|pos| scene_lock.get_nearest_px(*pos)); // closest to target node // Dijkstra's algorithm with 3 targets (yet to choose heuristic)
    for dest_node in dest_nodes { 
        path[dest_node] = usize::MAX - 2; 
        cost[dest_node] = 0.0; 
        set.push(dest_node);
    }
    
    let mut min_neighbor = usize::MAX - 1;
    while let Some(node) = set.pop() {
        min_neighbor = usize::MAX - 1;
        for neighbor in (*scene_lock).neighbors(node) {
            if cost[neighbor] < cost[min_neighbor] { // track back closest neighbor
                cost[node] = cost[min_neighbor] + scene_lock.height[min_neighbor]; // todo normalize map

                ball[node] = ball[neighbor];

                path[node] = min_neighbor;

                let min_neighbor = neighbor;
            }
            if ball[neighbor] == todo!("was not checked") {

                // PX:    1   2  // For space
                // BALL: 123 123
                // FMT: 0 = not checked 1 = checked
                ball_checked[node] *= u8::pow(2, ball[neighbor] as u32) * u8::pow(8, (min_neighbor % 2) as u32); // TODO fix up. Not the fastest way.
                
                todo!("Stopped here for the night")
            }
            if true { // Ball array which has it been checked against solution \/\/
               set.push(neighbor); // add all neighbors to hurrisric // ISSUE!!! loops back fix with arrays.
            }
        }
    }

    const start_node: usize = 0; // TODO calibrate
    let mut new_path = Vec::new();
    let mut node = start_node;
    while node != usize::MAX - 2 { 
        new_path.push(scene_lock.pos[node]);
        node = path[node];
    }

    drop(scene_lock);

    let mut path_lock = arc_path.lock().await; 
    *path_lock = Path {
        created: Instant::now(), 
        directions: new_path
    }
}

/// Handles Path Requests from Rio
pub(crate) async fn handle_path_request(path: Arc<Mutex<Path>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
                                // modified: created,
                                directions: Vec::new(),
                            }
                        };
                        socket.write(b"OK").await;
                    },
                    b"GetPath" => {
                        let path_lock = path.lock().await;
                        if let Err(e) = socket.write_all(&path_lock.serialize()).await {
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
