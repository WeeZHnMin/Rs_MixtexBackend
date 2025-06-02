//src/main.rs
#![windows_subsystem = "windows"]  // 放在最顶部
mod onnx_inference_module;
use onnx_inference_module::{process_image_with_padding, check_repetition};
mod state;
mod handlers;

use axum::{Router, routing::{post, get}, http::Method};
use std::sync::Arc;
use state::AppStore;
use handlers::{upload_image, stream_inference, final_decode, greet, bind_available_port};
use tower_http::cors::{CorsLayer, Any}; // ✅ 导入 CORS

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tokenizer_path = "./tokenizer/tokenizer.json";
    let model_folder = "./models";

    let app_store = Arc::new(AppStore::new(model_folder, tokenizer_path)?);

    // ✅ 添加 CORS 层，允许所有 origin/methods/headers
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);
    
    let app = Router::new()
        .route("/upload", post(upload_image))
        .route("/", get(greet))
        .route("/stream_inference", post(stream_inference))
        .route("/final_decode", post(final_decode))
        .with_state(app_store.clone())
        .layer(cors); // ✅ 添加 CORS Layer

    let (listener, addr) = bind_available_port(8000, 20).await?;

    axum::serve(listener, app)
        .await
        .unwrap();

    Ok(())
}