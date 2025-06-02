use std::net::SocketAddr;
use tokio::net::TcpListener;

pub async fn bind_available_port(start_port: u16, max_tries: u16) -> anyhow::Result<(TcpListener, SocketAddr)> {
    for port in start_port..(start_port + max_tries) {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        match TcpListener::bind(addr).await {
            Ok(listener) => return Ok((listener, addr)),
            Err(e) => {
                if e.kind() != std::io::ErrorKind::AddrInUse {
                    return Err(e.into());
                }
            }
        }
    }
    anyhow::bail!("No available port found in range {}-{}", start_port, start_port + max_tries - 1);
}