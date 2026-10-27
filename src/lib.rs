//! Bark 3.0 (Parakeet TDT 0.6B) — wgpu inference port; runs the whole family
//! (Pro, 3.0, 3.0 Plus, 3.0 Clean).
//!
//! [`AsrInference`] is the entry point: load a checkpoint once, then call
//! [`AsrInference::transcribe`] for each file. The result carries the reference
//! text, word timestamps, and subtitle segments (see [`TranscribeResult`]).
//!
//! Numeric contract:
//! - weights f16 packed two-per-u32 (low half first); activations and
//!   accumulation stay f32 (no f16 rounding between layers)
//! - mel feature extraction and TDT greedy decode on CPU
//!   (decoder weights f16, AVX2/F16C where available, scalar fallback)
//! - conv subsampling and the 24 conformer encoder layers on GPU, with a
//!   CPU-resident equivalent in `cpu` for the non-GPU backend
//! - the reference check is byte-exact transcription (see `verify.ps1`)

pub mod bark;
pub mod browser;
pub mod clock;
pub mod cpu;
pub mod encoder;
pub mod gpu;
pub mod inference;
pub mod mel;
pub mod parallel;
pub mod raw;
pub mod segment;
pub mod shaders;
pub mod sub;
pub mod sub_gpu;
pub mod tdt;
pub mod tdt_gpu;
pub mod transcript;
pub mod weights;
#[cfg(target_arch = "wasm32")]
pub mod weights_cache;
pub mod weights_io;

mod transcription;

pub use inference::{AsrInference, Backend, TranscribeOptions};
pub use segment::{Limits, Segment};
pub use transcript::{TranscribeResult, Word};

/// Model constants (from Underdog-Bark-3.0-Pro/config.json).
pub mod cfg {
    pub const HIDDEN: usize = 1024;
    pub const LAYERS: usize = 24;
    pub const HEADS: usize = 8;
    pub const HEAD_DIM: usize = HIDDEN / HEADS; // 128
    pub const FFN: usize = 4096;
    pub const CONV_K: usize = 9;
    pub const SUB_CHANNELS: usize = 256;
    pub const SUB_FACTOR: usize = 8;
    pub const MEL_BINS: usize = 128;
    pub const DEC_HIDDEN: usize = 640;
    pub const VOCAB: usize = 8193;
    pub const N_DUR: usize = 5; // joint head is VOCAB + 5
    pub const BLANK: u32 = 8192;
    pub const UNK: u32 = 0;
    pub const PAD: u32 = 2;
    pub const EOS: u32 = 3;
    pub const LN_EPS: f32 = 1e-5;
    pub const MAX_SYMBOLS: u32 = 10;
    pub const DURATIONS: [u32; 5] = [0, 1, 2, 3, 4];
}
