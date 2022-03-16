use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::time::Instant;
use crate::scene::{Scene, Target};

pub(crate) struct Path {
    pub(crate) created: Instant,
    pub(crate) modified: Instant,
    pub(crate) directions: Vec<u8>,
}

/// Modifies the Path with the new scene information
pub(crate) async fn modify_path(path: Arc<Mutex<Path>>, target_queue: Arc<Mutex<Vec<Target>>>, scene: Arc<Mutex<Scene>>) {
    scene.readable().await;
    let mut scene_lock = scene.lock().unwrap();
    
    let mut set = BTreeSet::new();
    let mut path: [i32; 224 * 224] = [-1; 224 * 224]; // -1 = undefined -2 = target
    let mut cost: [u32; 224 * 224] = [u32::MAX; 224 * 224]; // Cost to travel from a point

    let dest_node = scene_lock.balls[..3]; // closest to target node // Dijkstra's algorithm with 3 targets
    path[dest_node[0]] = -2; path[dest_node[1]] = -2; path[dest_node[2]] = -2;
    cost[dest_node[0]] = 0; cost[dest_node[1]] = 0; cost[dest_node[2]] = 0; 
    set.insert(dest_node);
    let mut min_neighbor = -1;
    for node in set {
        min_neighbor = -1;
        for neighbor in scene.get_node().neighbors() {
            if cost[neighbor] < cost[min_neighbor] {
                let min_neighbor = neighbor;
                cost[node] = cost[min_neighbor] + scene_lock.height[min_neighbor]; // todo normalize map
                path[node] = min_neighbor;
            }
            set.remove(neighbor);
        }
    }

    const start_node: i32 = todo!(); // TODO calibrate
    let new_path = Vec::new();
    let node = start_node;
    while node != -2 { 
        new_path.push(scene_lock.pos2d[node]);
        node = path[node]
    }

    drop(scene_lock);

    path.writable().await;
    let path_lock = path.lock().unwrap(); 
    path_lock = Path {
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
                                modified: created,
                                directions: Vec::new(),
                            }
                        };
                        socket.write(b"OK").await;
                    },
                    b"GetPath" => {
                        let path_lock = path.lock().await;
                        if let Err(e) = socket.write_all(&path_lock.directions[..]).await {
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
