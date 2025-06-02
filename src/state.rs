use std::sync::{Arc, Mutex};
use crate::onnx_inference_module::{OrtInferenceSession, TemporaryData};

pub struct AppStore {
    pub onnx_session: Arc<OrtInferenceSession>,
    pub temporary_data: Arc<Mutex<TemporaryData>>,
}

impl AppStore {
    pub fn new(model_folder: &str, tokenizer_path: &str) -> anyhow::Result<Self> {
        let onnx_session = Arc::new(OrtInferenceSession::new(model_folder, tokenizer_path)?);
        let temporary_data = Arc::new(Mutex::new(TemporaryData::new()));
        Ok(Self { onnx_session, temporary_data })
    }
}