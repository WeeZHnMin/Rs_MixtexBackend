
<p align="center">
  <img src="icon.ico" width="450" height="450">
</p>

## 项目介绍 | Project Introduction

本项目基于 Rust 语言开发，使用 `ort` 与 `tokenizers` 模块实现 OCR 推理，后端框架采用 [Axum](https://github.com/tokio-rs/axum)。

推理模型由两个部分组成：Encoder 为 SwinModel，Decoder 为 GPT-2 模型。模型托管在 Hugging Face 仓库：[https://huggingface.co/MixTex/base\_ZhEn](https://huggingface.co/MixTex/base_ZhEn)

> This project is developed in **Rust**, using the `ort` and `tokenizers` crates to implement OCR inference. The backend is built with **Axum** framework.
> The inference model consists of two components:
>
> * **Encoder**: SwinModel
> * **Decoder**: GPT-2
>   The model is hosted on Hugging Face: [https://huggingface.co/MixTex/base\_ZhEn](https://huggingface.co/MixTex/base_ZhEn)

---

## 下载模型 | Pulling the Model Files

请在项目根目录中打开 PowerShell，然后执行以下命令：

```powershell
mkdir backend\models
Invoke-WebRequest -Uri "https://huggingface.co/wzmmmm/_wmzmz/resolve/main/encoder_model.onnx" -OutFile "models/encoder_model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/wzmmmm/_wmzmz/resolve/main/decoder_model.onnx" -OutFile "models/decoder_model.onnx"
```

> Run the above commands in **PowerShell** to download the model files into the local `models/` directory.

如果无法使用 PowerShell 下载，请手动访问以下链接下载文件：

* 🔗 [https://huggingface.co/wzmmmm/\_wmzmz/tree/main](https://huggingface.co/wzmmmm/_wmzmz/tree/main)

下载以下两个文件，并放入项目的 `models/` 目录中（如该目录不存在请自行创建）：

* `encoder_model.onnx`
* `decoder_model.onnx`

> If you cannot download via PowerShell, please download them manually from the link above and place them under the `models/` directory.

---

## 启动服务 | Running the Project

### 构建项目 | Build

```cmd
cargo build
```

### 启动项目 | Run

```cmd
cargo run
```

当你看到程序监听在 `localhost:8000`，说明模型已经加载完成并开始提供推理服务。

> After running the program, if you see the model listening at `localhost:8000`, it means the service is running successfully and the model is ready.
