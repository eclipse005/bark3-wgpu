# Bark 3.0 wgpu

**Bark 3.0 speech recognition in Rust with wgpu.**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/bark3-pro-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/bark3-pro-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/bark3-pro-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/bark3-pro-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/bark3-pro-wgpu/build.yml?branch=main&style=flat-square&label=build)](https://github.com/eclipse005/bark3-pro-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/bark3-pro-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=windows)](#install) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#install) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#install)

**English** · [简体中文](README.zh-CN.md)

### Features

* 🦀 Pure Rust, single binary
* 🎮 GPU acceleration with wgpu (Vulkan / D3D12 / Metal / GL)
* ⚡ CPU backend — no adapter, no VRAM, works on machines without a GPU
* 🖥️ Windows / macOS / Linux
* 📦 Offline local inference
* 🎙️ Bark 3.0 Pro / Plus / 3.0 / Clean (English)
* 🔢 Low-bit weights — a 163 MB release runs in ~380 MB of VRAM
* ⏱️ Word-level and subtitle-level timestamps in one JSON document
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
bark3-pro-wgpu = { git = "https://github.com/eclipse005/bark3-pro-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/bark3-pro-wgpu.git
cd bark3-pro-wgpu
cargo build --release        # target/release/bark3
```

### Model download

Weights are **not** included in this repository (rights remain with the original authors, CC-BY-4.0). Download a release and pass the directory to `--model`:

| Model | Download | Notes |
|---|---:|---|
| [Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) | 1.25 GB | Full precision, best accuracy |
| [Underdog-Bark-3.0-Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus) | 180 MB | Low-bit, closest to Pro |
| [Underdog-Bark-3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) | 163 MB | Low-bit, smallest download |
| [Underdog-Bark-3.0-Clean](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Clean) | 180 MB | Low-bit, removes fillers and false starts |

Pro ships `model.safetensors` and works as downloaded. The other three ship a compressed `model.bark` container; convert it once into this crate's low-bit layout:

```bash
pip install numpy constriction safetensors
python tools/encode_lowbit.py model.bark ../models/Underdog-Bark-3.0-Plus \
    --reference ../models/Underdog-Bark-3.0-Pro
```

The converter keeps the 4-bit codes and per-row level tables, so the low-bit
weights land at ~380 MiB of VRAM instead of 1.2 GiB, and it records the release
name plus the container's SHA-256 in `bark3.json`.

### Quick Start

```bash
bark3 --model ../models/Underdog-Bark-3.0-Pro transcribe ./audio.wav
```

The transcript goes to **stdout as one JSON document**; device, timings and the
detected model stay on stderr:

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

`jq -r .text` gives the plain transcript; `.segments` is what a subtitle writer
needs; `.words` is the word timeline.

| Option | Description |
|--------|-------------|
| `--model <dir>` | Checkpoint directory (or the `BARK3_MODEL` environment variable). Required: the binary has no built-in path, and it checks the directory before loading |
| `BARK3_CPU_ENC=1` | Run the CPU backend (no adapter is created) |

Input audio must be 16 kHz mono WAV. `BARK3_CPU_DEC=1` selects the CPU decoder for A/B work.

### Library

```rust
use bark3_pro_wgpu::{AsrInference, Backend, TranscribeOptions};

let asr = AsrInference::load("models/Underdog-Bark-3.0-Pro", Backend::best())?;
let out = asr.transcribe("audio.wav", TranscribeOptions::default())?;

println!("{}", out.text);
for segment in &out.segments {
    println!("{:6.2} {:6.2}  {}", segment.start, segment.end, segment.text);
}
```

A model is loaded once and reused; `transcribe` takes `&self`, so one instance can
be shared. Also available:

* In-memory audio — `transcribe_samples(&samples_16k, opts)`; no WAV, no file
* Pick the compute path — `Backend::Auto` (default: GPU, CPU fallback), `Backend::Cpu`, `Backend::Gpu`; `AsrInference::backend()` and `device_description()` report what ran
* Subtitle segmentation — `TranscribeOptions::default().with_segment_gap(1.0).with_segment_max_chars(84).with_segment_max_seconds(7.0)`
* What was loaded — `AsrInference::model_info()` (release, layout, parameter count, architecture)
* JSON — `TranscribeResult::to_json()`

`TranscribeResult` carries `text` (the reference rendering, byte-exact against the
official pipeline), `words` (`text`/`start`/`end` per word), `segments` (cues built
from words), `tokens` (raw decoder output) and `duration`. Timestamps come from the
TDT decoder's own frame+duration output on an 80 ms grid — decode timing, not
forced alignment.

See `cargo doc` for the full API.

### Why wgpu?

Instead of relying on CUDA, ROCm or other vendor-specific runtimes, this project
uses **wgpu** as a unified GPU abstraction (Vulkan, D3D12, Metal, GL). Weights are
stored as f16 or as 4-bit codes expanded inside the GEMM load stage, so the same
shaders run on any of those backends. When no adapter can be opened, the
CPU-resident encoder takes over with vectorised kernels and no VRAM at all.

### Project Status

🚧 **Active development**

The GPU and CPU paths reproduce the reference pipeline byte for byte on the
English evaluation sets used during development. Hardware coverage beyond the
development machine (one Vulkan GPU, one x86 CPU) is not yet exercised.

### Related

* [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) — the model family
* [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — the upstream architecture
* [wgpu](https://github.com/gfx-rs/wgpu)

### License

MIT for this implementation. Model weights keep their own terms (CC-BY-4.0, with
attribution as listed in the model repositories).

This repository is an **independent Rust inference implementation** for running
the officially released Bark 3.0 weights — not an official Underdog release and
not affiliated with the original authors.
