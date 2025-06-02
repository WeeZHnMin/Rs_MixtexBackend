use axum::{extract::State, response::sse::{Event, Sse}};
use std::{convert::Infallible, pin::Pin, sync::Arc};
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;

use crate::state::AppStore;
use crate::check_repetition;

pub async fn stream_inference(
    State(app_store): State<Arc<AppStore>>,
) -> Sse<Pin<Box<dyn futures::Stream<Item = Result<Event, Infallible>> + Send>>> {
    let temp_data = Arc::clone(&app_store.temporary_data);
    
    // 1. 先获取并克隆图像数据
    let input_image = {
        let guard = temp_data.lock().unwrap();
        match guard.get_image() {
            Some(img) => img.clone(),
            None => {
                return Sse::new(tokio_stream::empty::<Result<Event, Infallible>>().boxed());
            }
        }
    }; // MutexGuard在这里被释放

    let (tx, rx) = tokio::sync::mpsc::channel(16);
    let onnx_session = Arc::clone(&app_store.onnx_session);
    let temp_data = Arc::clone(&temp_data); // 克隆一份用于发送到新任务

    tokio::spawn(async move {
        let tokenizer = onnx_session.get_tokenizer();
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(30000);
        let bos_token_id = tokenizer.token_to_id("<s>").unwrap_or(0);
        let max_len = 512;

        let (decoder_outputs, next_token_id, encoder_hidden_states) = match onnx_session.init_inference(input_image) {
            Ok(res) => res,
            Err(e) => {
                let _ = tx.send(format!("初始化推理失败: {:?}", e)).await;
                return;
            }
        };
        let next_token = tokenizer.decode(&vec![next_token_id], true).unwrap_or_default();
        let _ = tx.send(next_token.clone()).await;
        let mut token_id_array = vec![bos_token_id, next_token_id];

        let mut decoder_inputs = decoder_outputs;
        let mut encoder_input = encoder_hidden_states;
        let mut input_token_id = next_token_id;

        for _i in 0..max_len {
            let (decoder_outputs, next_token_id, encoder_hidden_states) =
                match onnx_session.single_inference(decoder_inputs, input_token_id, encoder_input) {
                    Ok(res) => res,
                    Err(e) => {
                        let _ = tx.send(format!("单次推理失败: {:?}", e)).await;
                        break;
                    }
                };

            token_id_array.push(next_token_id);
            let is_stop = check_repetition(&token_id_array, 10);
            if is_stop {
                let _ = tx.send("\n\n推理异常，停止推理".to_string()).await;
                break;
            }

            let next_token = tokenizer.decode(&vec![next_token_id], true).unwrap_or_default();
            let _ = tx.send(next_token.clone()).await;
            if next_token_id == eos_token_id {
                break;
            }
            decoder_inputs = decoder_outputs;
            encoder_input = encoder_hidden_states;
            input_token_id = next_token_id;
        }
        // 将 token_id_array 存储到临时数据中
        // 在发送消息之前先完成数据更新
        // 将锁的获取和使用放在最小范围内
        {
            if let Ok(mut guard) = temp_data.lock() {
                guard.set_token_id_array(token_id_array);
            }
        } // 锁在这里被释放

        // 3. 错误消息的发送移到锁释放之后
        if temp_data.lock().is_err() {
            let _ = tx.send("临时数据锁定失败".to_string()).await;
        }
    });


    let stream = ReceiverStream::new(rx)
        .map(|token| Ok(Event::default().data(token)))
        .boxed();
    Sse::new(stream)
}