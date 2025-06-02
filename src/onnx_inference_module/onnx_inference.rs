//src/onnx_inference_copy.rs
use anyhow::Ok;
use ort::session::Session;
use ort::session::SessionBuilder;
use ort::GraphOptimizationLevel;
use ort::LoggingLevel;
use ort::Environment;
use ort::Value;
use ort::tensor::OrtOwnedTensor;

use tokenizers::Tokenizer;
use ndarray::{Array, ArrayBase, OwnedRepr, Dim, ArrayD, IxDyn, CowArray, IxDynImpl};
use std::path::PathBuf;

use crate::process_image_with_padding;

pub struct OrtInferenceSession {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer
}


impl OrtInferenceSession {
    pub fn new(model_folder: &str, _tokenizer_path: &str) -> anyhow::Result<Self> {
        let environment = Environment::builder()
            .with_name("mixtex_environment")
            .with_log_level(LoggingLevel::Verbose)
            .build()?
            .into_arc();

        // Create a temporary instance
        let encoder_path = PathBuf::from(model_folder).join("encoder_model.onnx");
        let encoder_session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_inter_threads(1)?
            .with_model_from_file(encoder_path)?;

        let decoder_path = PathBuf::from(model_folder).join("decoder_model.onnx");
        let decoder_session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_inter_threads(1)?
            .with_model_from_file(decoder_path)?;
        let tokenizer = Tokenizer::from_file(_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer
        })
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn encode_image(&self, input_image: image::DynamicImage) -> anyhow::Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> {
        let image_data = process_image_with_padding(input_image, "processed_image.png")?;

        // Step 2: 转为动态维度 (IxDyn)
        let dyn_image: ArrayD<f32> = image_data.into_dyn();

        // ✅ Step 3: 构建 CowArray（关键）
        let cow_array: CowArray<f32, IxDyn> = CowArray::from(dyn_image);

        // 4. 创建 ONNX 输入
        let allocator = self.encoder_session.allocator();
        let input = Value::from_array(allocator, &cow_array)?;

        let outputs = self.encoder_session.run(vec![input])?;
        let outpu_ort: OrtOwnedTensor<'_, f32, Dim<IxDynImpl>> = outputs[0].try_extract::<f32>()?;
        let output_array = outpu_ort.view().to_owned();
        // println!("Encoder outputs shape: {:?}", outpu_ort.view().shape());

        Ok(output_array)
    }

    
    // pub fn image_inference(&self, input_image: image::DynamicImage) -> anyhow::Result<String> {
    //     let eos_token_id: u32 = self.tokenizer.token_to_id("</s>").unwrap();
    //     let bos_token_id: u32 = self.tokenizer.token_to_id("<s>").unwrap();

    //     let max_len = 512;
    //     let (decoder_outputs, next_token_id, encoder_output) = self.init_inference(input_image)?;

    //     let mut token_id_arr = vec![bos_token_id, next_token_id];
    //     let mut decoder_inputs = decoder_outputs;
    //     let mut encoder_input = encoder_output;
    //     let mut input_token_id = next_token_id;

    //     for _i in 0..max_len {
    //         let (decoder_outputs, next_token_id, encoder_hidden_states) = self.single_inference(decoder_inputs, input_token_id, encoder_input)?;
    //         token_id_arr.push(next_token_id);
    //         if next_token_id == eos_token_id {
    //             break;
    //         }
    //         decoder_inputs = decoder_outputs;
    //         encoder_input = encoder_hidden_states;
    //         input_token_id = next_token_id;
    //     }
    //     let decode_res = self.tokenizer.decode(&token_id_arr, true)
    //         .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

    //     Ok(decode_res)
    // }

    pub fn single_inference(&self, input_vec: Vec<Value<'static>>, input_token_value: u32, encoder_hidden_states: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> anyhow::Result<(Vec<Value<'static>>, u32, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>)> {
        let input_ids = Array::from_shape_vec(IxDyn(&[1, 1]), vec![input_token_value as i64]).to_owned()?;
        let cow_input_ids = CowArray::from(input_ids);
        let encoder_hidden_states_copy = encoder_hidden_states.to_owned();
        let cow_encoder_hidden_states = CowArray::from(encoder_hidden_states);
        let allocator = self.decoder_session.allocator();
        
        let input  = Value::from_array(allocator, &cow_input_ids)?;
        let encoder_input = Value::from_array(allocator, &cow_encoder_hidden_states)?;

        let unsafe_input = unsafe {
            std::mem::transmute::<Value<'_>, Value<'static>>(input)
        };
        let unsafe_encoder_input: Value<'static> = unsafe {
            std::mem::transmute::<Value<'_>, Value<'static>>(encoder_input)
        };

        let mut decoder_inputs = vec![unsafe_input, unsafe_encoder_input];
        decoder_inputs.extend(input_vec);

        let mut decoder_outputs: Vec<Value<'static>> = self.decoder_session.run(decoder_inputs)?;
        let logits_output = decoder_outputs[0].try_extract::<f32>()?;
        let next_token_id: u32 = logits_output
            .view()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        decoder_outputs.remove(0); // Remove logits output
        Ok((decoder_outputs, next_token_id, encoder_hidden_states_copy))
    }

    pub fn init_inference(&self, input_image: image::DynamicImage) -> anyhow::Result<(Vec<Value<'static>>, u32, ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>)> {
        let encoder_hidden_states = self.encode_image(input_image)?;
        let bos_token_id: u32 = self.tokenizer.token_to_id("<s>").unwrap();

        let num_layers = 6;
        let num_heads = 12;
        let seq_len = 1;
        let head_dim = 64;
        let batch_size = 1;

        let input_ids = {
            Array::from_shape_vec(
                IxDyn(&[batch_size, seq_len]),
                vec![bos_token_id as i64],
            )?
        };

        let past_tensor = ArrayD::<f32>::zeros(IxDyn(&[batch_size, num_heads, 0, head_dim]));
        let mut past_keys = Vec::with_capacity(num_layers);
        let mut past_values = Vec::with_capacity(num_layers);
        for _i in 0..num_layers {
            past_keys.push(CowArray::from(past_tensor.clone()));
            past_values.push(CowArray::from(past_tensor.clone()));
        }

        let cow_input_ids = CowArray::from(input_ids);
        let cow_encoder_hidden_states = CowArray::from(encoder_hidden_states.clone());
        let encoder_hidden_states_copy = encoder_hidden_states.to_owned();
        let allocator = self.decoder_session.allocator();
        let input = Value::from_array(allocator, &cow_input_ids)?;
        let encoder_input = Value::from_array(allocator, &cow_encoder_hidden_states)?;
        
        let mut decoder_inputs = Vec::new();
        decoder_inputs.push(input);
        decoder_inputs.push(encoder_input);

        for i in 0..num_layers {
            decoder_inputs.push(Value::from_array(allocator, &past_keys[i])?);
            decoder_inputs.push(Value::from_array(allocator, &past_values[i])?);
        }
        let mut decoder_outputs: Vec<Value<'static>> = self.decoder_session.run(decoder_inputs)?;
        let logits_output = decoder_outputs[0].try_extract::<f32>()?;
        let next_token_id: u32 = logits_output
            .view()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        decoder_outputs.remove(0); // Remove logits output
        Ok((decoder_outputs, next_token_id, encoder_hidden_states_copy))
    }
}