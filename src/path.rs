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
    todo!();
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
