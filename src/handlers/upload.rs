use axum::{extract::{Multipart, State}, response::IntoResponse, http::StatusCode};
use std::sync::Arc;
use crate::state::AppStore;

pub async fn upload_image(
    State(app_store): State<Arc<AppStore>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            let data = field.bytes().await.unwrap();
            match image::load_from_memory(&data) {
                Ok(img) => {
                    // 注意这里要 lock
                    if let Ok(mut temp_data) = app_store.temporary_data.lock() {
                        temp_data.set_image(img);
                        return (StatusCode::OK, "图片上传成功").into_response();
                    } else {
                        return (StatusCode::INTERNAL_SERVER_ERROR, "数据锁定失败").into_response();
                    }
                }
                Err(e) => {
                    return (StatusCode::BAD_REQUEST, format!("图片解码失败: {}", e)).into_response();
                }
            }
        }
    }
    (StatusCode::BAD_REQUEST, "没有找到图片字段").into_response()
}
