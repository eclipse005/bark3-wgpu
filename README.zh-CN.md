# Bark 3.0 wgpu

**基于 wgpu 的 Bark 3.0 语音识别（Rust 实现）。**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/bark3-pro-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/bark3-pro-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/bark3-pro-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/bark3-pro-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/bark3-pro-wgpu/build.yml?branch=main&style=flat-square&label=build)](https://github.com/eclipse005/bark3-pro-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/bark3-pro-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=windows)](#安装) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#安装) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#安装)

[English](README.md) · **简体中文**

### 特性

* 🦀 纯 Rust，单二进制
* 🎮 wgpu GPU 加速（Vulkan / D3D12 / Metal / GL）
* ⚡ CPU 后端：不创建适配器、不占显存，无显卡机器也能跑
* 🖥️ Windows / macOS / Linux
* 📦 离线本地推理
* 🎙️ Bark 3.0 Pro / Plus / 3.0 / Clean（英文）
* 🔢 低比特权重：163 MB 的发布版只占约 380 MB 显存
* ⏱️ 词级与字幕级时间戳，一个 JSON 文档全给
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
bark3-pro-wgpu = { git = "https://github.com/eclipse005/bark3-pro-wgpu.git" }
```

或从源码构建：

```bash
git clone https://github.com/eclipse005/bark3-pro-wgpu.git
cd bark3-pro-wgpu
cargo build --release        # target/release/bark3
```

### 模型下载

权重**不在**本仓库内（版权归原作者，CC-BY-4.0）。下载后把目录传给 `--model` 即可，不需要任何转换：

| 模型 | 下载 | 布局 | 说明 |
|---|---:|---|---|
| [Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) | 1.25 GB | `model.safetensors`（bf16） | 全精度，精度最好 |
| [Underdog-Bark-3.0-Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus) | 180 MB | `model.bark` | 低比特，最接近 Pro |
| [Underdog-Bark-3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) | 163 MB | `model.bark` | 低比特，下载最小 |
| [Underdog-Bark-3.0-Clean](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Clean) | 180 MB | `model.bark` | 低比特，自动去掉口头语与重复 |

压缩版把 encoder 线性层存成「每行 fp16 电平 + 4-bit 码」。加载器直接读这个容器、把码交给
GEMM 的装载段展开，所以 163 MB 的下载只占约 380 MiB 显存（f16 展开要 1.2 GiB）。布局从头部
自动识别，架构不匹配的 checkpoint 会在上传任何权重之前被拒绝。

### 快速上手

```bash
bark3 --model ../models/Underdog-Bark-3.0-Pro transcribe ./audio.wav
```

转录结果是**一个 JSON 文档**走 stdout；设备、耗时和识别出的模型走 stderr：

```
[model] Underdog-Bark-3.0-Pro (bf16, 627M weights, 723 tensors)
[load] gpu backend in 3.4s
[time] mel 0.026s sub 0.146s enc 1.219s dec 0.162s | audio 176.56s RTFx=113.7x | tokens 895 words 410 segments 62
```

```json
{
  "text": "Alright flippers, welcome everyone into this crash course ...",
  "duration": 15.0,
  "segments": [
    {"start": 1.67, "end": 6.15, "text": "Alright flippers, welcome everyone into this crash course about our new fifteen"}
  ],
  "words": [
    {"text": "Alright", "start": 1.67, "end": 2.07}
  ]
}
```

`jq -r .text` 取纯文本；`.segments` 直接可以写字幕；`.words` 是词级时间轴。

| 参数 | 说明 |
|------|------|
| `--model <dir>` | 权重目录（或环境变量 `BARK3_MODEL`）。**必填**：二进制没有内置路径，加载前会校验目录 |
| `BARK3_CPU_ENC=1` | 走 CPU 后端（不创建适配器） |

输入必须是 16 kHz 单声道 WAV。`BARK3_CPU_DEC=1` 可切到 CPU 解码做 A/B。

### 作为库使用

```rust
use bark3_pro_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("models/Underdog-Bark-3.0-Pro", Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;

println!("{}", out.text);
for segment in &out.segments {
    println!("{:6.2} {:6.2}  {}", segment.start, segment.end, segment.text);
}
```

模型只加载一次，`transcribe` 取 `&self`，一个实例可共享。另有：

* 内存音频 —— `transcribe_samples(&samples_16k, opts)`，不需要 WAV 文件
* 选择计算路径 —— `Backend::Auto`（默认：有 GPU 就用，打不开回退 CPU）、`Backend::Cpu`、`Backend::Gpu`；`backend()` / `device_description()` 报告实际用的
* 字幕分段策略 —— `TranscribeOptions::default().with_segment_gap(1.0).with_segment_max_chars(84).with_segment_max_seconds(7.0)`
* 加载了什么 —— `AsrInference::model_info()`（发布名、布局、参数量、架构）
* JSON —— `TranscribeResult::to_json()`

`TranscribeResult` 含 `text`（参考渲染，与官方管线逐字节一致）、`words`（每词
`text`/`start`/`end`）、`segments`（由词合成的字幕段）、`tokens`（原始解码输出）与
`duration`。时间戳来自 TDT 解码器自己的 frame+duration，**80 ms 栅格**——是解码时间，
不是强制对齐。

完整 API 见 `cargo doc`。

### 浏览器（WebAssembly）

`wasm/` 是浏览器绑定：同一套引擎编译到 `wasm32-unknown-unknown`，走 WebGPU（打不开时回退
CPU encoder）。

```bash
rustup target add wasm32-unknown-unknown
cargo build --release -p bark3-wasm --target wasm32-unknown-unknown
# 再用 wasm-bindgen 处理 target/wasm32-unknown-unknown/release/bark3_wasm.wasm
```

```js
import init, { BarkAsr } from "./bark3_wasm.js";
await init();

const bytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
const asr = await BarkAsr.create(bytes, "Underdog-Bark-3.0-Plus", "auto");

// 16 kHz 单声道 f32，加上 VAD 给出的切点（每段静音的中点）：
const out = JSON.parse(asr.transcribeChunked(samples16k, cuts));
console.log(out.text, out.words.length);
```

返回的就是 CLI 那份文档——`text` / `duration` / `segments` / `words`，词时间是整段录音的
时间轴，所以下游断句、翻译都不用改。`transcribe(samples)` 是整段一次解码；
`transcribe_chunked` 对每个语音段各留 0.2 s 上下文，并保证每个词只归一个段。

CI 会构建 wasm 目标。端到端验证需要 WebGPU，实用做法是开一个页面喂固定 WAV，把 JSON 和
同一文件的 `bark3 transcribe` 输出逐字节比对：代码同源、只有设备不同，输出必须一致。

### 为什么选择 wgpu？

本项目不依赖 CUDA / ROCm 等厂商专属运行时，而是用 **wgpu** 作为统一的 GPU 抽象
（Vulkan、D3D12、Metal、GL）。权重以 f16 或 4-bit 码 + 行内电平表存储，dequant 发生在
GEMM 的装载段，因此同一套 shader 能跑在这些后端上；打不开适配器时切到 CPU encoder，
不占显存。

### 项目状态

🚧 **积极开发中**

GPU 与 CPU 两条路径在开发所用的英文评测集上都与官方管线逐字节一致；除开发机（一块
Vulkan 显卡 + 一颗 x86 CPU）以外的硬件尚未验证。

### 相关

* [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) —— 模型家族
* [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) —— 上游架构
* [wgpu](https://github.com/gfx-rs/wgpu)

### 许可证

本实现使用 MIT。模型权重沿用模型仓库的条款（CC-BY-4.0，署名要求见各模型仓库的 NOTICE）。

本仓库是**独立的 Rust 推理实现**，用于运行官方发布的 Bark 3.0 权重，不是 Underdog 官方
发布，也与原作者无隶属关系。
