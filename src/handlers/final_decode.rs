use axum::{extract::State, response::IntoResponse, http::StatusCode};
use std::sync::Arc;
use crate::state::AppStore;

pub async fn final_decode(
    State(app_store): State<Arc<AppStore>>,
) -> impl IntoResponse {
    let temp_data = app_store.temporary_data.lock().unwrap();
    let token_id_array = temp_data.token_id_array();

    if token_id_array.is_empty() {
        return (StatusCode::BAD_REQUEST, "貌似还没有上传图片").into_response();
    }

    let tokenizer = app_store.onnx_session.get_tokenizer();
    let decoded_text = tokenizer.decode(token_id_array, true).unwrap_or_default();

    (StatusCode::OK, format!("{}", decoded_text)).into_response()
}