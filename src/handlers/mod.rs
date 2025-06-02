mod upload;
mod stream;
mod final_decode;
mod bind_port;

pub use upload::upload_image;
pub use stream::stream_inference;
pub use final_decode::final_decode;
pub use bind_port::bind_available_port;

pub async  fn greet() -> &'static str {
    "Hello, welcome to the ONNX inference server!"
}