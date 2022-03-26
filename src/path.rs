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
    fn serialize(&self) -> Vec<u8> {
        self.directions.iter().map(|(x, y)| [x.to_be_bytes(), y.to_be_bytes()].concat()).flatten().collect()
    }
}

/// Modifies the Path with the new scene information
pub(crate) async fn modify_path(arc_path: Arc<Mutex<Path>>, target_queue: Arc<Mutex<Vec<Target>>>, scene: Arc<Mutex<Scene>>) {
    let mut scene_lock = scene.lock().await;

    let mut set = Vec::new(); // TODO use TreeSet (first & last requires nightly)
    let mut path: [usize; 224 * 224] = [usize::MAX - 1; 224 * 224]; // MAX-1 = undefined MAX-2 = target (I do not need u32 size)
    let mut cost: [f32; 224 * 224] = [f32::MAX; 224 * 224]; // Cost to travel from a point

    // Commented out optimization
    //let mut ball_checked: [u8; 224 * 112] = [0; 224 * 112];
    
    let mut ball: [u8; 224 * 56] = [0; 224 * 56]; // Use in optimizations and UI

    let ball_poss = &scene_lock.balls[..3];
    let dest_nodes = ball_poss.iter().map(|pos| scene_lock.get_nearest_px(*pos)); // closest to target node // Dijkstra's algorithm with 3 targets (yet to choose heuristic)
    for dest_node in dest_nodes { 
        path[dest_node] = usize::MAX - 2; 
        cost[dest_node] = 0.0; 
        set.push(dest_node);

        // Part of commented out optimizations
        // PX:    1    2  // For space
        // BALL: 321_ 321_  // The last value set to one
        // FMT: 0 = not checked 1 = checked
        //ball_checked[dest_node] = 17;
    }
    
    let mut min_neighbor_cost = f32::MAX;
    while let Some(node) = set.pop() {
        //best_ball = ball[node]; // ball 0 has 100% been checked
        min_neighbor_cost = cost[node];
        for (cn, neighbor) in (*scene_lock).neighbors(node).into_iter().enumerate() {
            if cost[neighbor] == f32::MAX {
                set.push(neighbor);
            } else {
                let neighbor_cost = cost[neighbor] + scene_lock.connections[node][cn] + f32::abs(scene_lock.height[node] - scene_lock.height[neighbor]);
                if neighbor_cost < min_neighbor_cost {
                    min_neighbor_cost = cost[neighbor];

                    cost[node] = neighbor_cost;

                    // best_ball = ball[neighbor];
                    ball[node] = ball[neighbor];

                    path[node] = neighbor;
                } 
            }

            // PX:    1    2  // For space
            // BALL: 321_ 321_  // The last value set to one
            // FMT: 0 = not checked 1 = checked
            //ball_checked[node] &= (1 << ball[neighbor]) << ((node >> 1 << 1 ^ node) << 2 /* mul 4 */); 
            // I May be able to use for optimizations
        }
        //ball[node] = best_ball; // Commented out optimizations.

        // TODO: Certain things could be optimized
        // Commented out: skip nodes with ball[neighbor] = ball[node] // Could obscure paths (Already not perfect)
        // skip already checked nodes. // Could obscure paths (Already not perfect)
        // ...
        // Done: min_neighbor_cost could default to cost[node] rather than f32::MAX
        // Considering: Positional Optimizations
        for (cn, neighbor) in (*scene_lock).neighbors(node).into_iter().enumerate() {
            if cost[neighbor] > min_neighbor_cost + scene_lock.connections[node][cn] + f32::abs(scene_lock.height[node] - scene_lock.height[neighbor]) {
                set.push(neighbor);
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
