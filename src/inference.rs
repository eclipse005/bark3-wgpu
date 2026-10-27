//! The library entry point: load once, transcribe many.
//!
//! ```no_run
//! use bark3_wgpu::{AsrInference, Backend, TranscribeOptions};
//!
//! # fn main() -> anyhow::Result<()> {
//! let asr = AsrInference::load("models/Underdog-Bark-3.0-Pro", Backend::best())?;
//! let out = asr.transcribe("clip.wav", TranscribeOptions::default())?;
//! println!("{}", out.text);
//! for segment in &out.segments {
//!     println!("{:6.2} {:6.2}  {}", segment.start, segment.end, segment.text);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`AsrInference`] owns the model and takes `&self`, so one instance can be
//! shared. [`Backend`] picks the compute path: the wgpu encoder, the
//! CPU-resident encoder, or the automatic choice between them. The mel front end
//! and the vocabulary are bundled, so the only input is the checkpoint
//! directory — `model.safetensors` plus `config.json`.
//!
//! Internals (`gpu`, `tdt`, `sub_w`, `cpu`, `mel_fe`) are public only because
//! the diagnostic binaries are separate crates that drive the stages directly.
//! A library user needs [`AsrInference::load`] and
//! [`AsrInference::transcribe`] / [`AsrInference::transcribe_samples`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cpu::CpuEncoder;
use crate::encoder::Encoder;
use crate::gpu::{DeviceSelector, Gpu};
use crate::mel::{MelFrontend, read_wav};
use crate::segment::Limits;
use crate::sub::SubsamplingWeights;
use crate::sub_gpu::SubGpu;
use crate::tdt::TdtDecoder;
use crate::tdt_gpu::GpuTdtDecoder;
use crate::transcript::TranscribeResult;
use crate::weights::WeightSource;

/// The NeMo mel filterbank the reference front end uses, bundled as f32 so the
/// checkpoint directory is the only input.
const MEL_FILTERS: &[u8] = include_bytes!("../data/mel_filters.f32");

/// Sample rate the model was trained on. Input audio must already be 16 kHz
/// mono; [`AsrInference::transcribe`] rejects anything else.
pub const SAMPLE_RATE: u32 = 16_000;

/// Which compute path a [`AsrInference`] runs.
///
/// * [`Self::Auto`] (default) — the wgpu backend when it can be built, the CPU
///   backend when it cannot. That fallback prints why, because 80x real time on
///   a GPU versus 28x on the CPU should never happen silently.
/// * [`Self::Gpu`] — require wgpu. No adapter or a failed init is an error.
/// * [`Self::Cpu`] — the CPU-resident encoder. No adapter is created, so this
///   runs on machines without a GPU.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl Backend {
    /// The best backend this machine can offer — i.e. [`Self::Auto`]. The name
    /// exists so a caller can say "give me the good one" without hard-coding a
    /// variant.
    #[must_use]
    pub const fn best() -> Self {
        Backend::Auto
    }

    /// Short human label, for logs and for asserting what got picked.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

/// Per-request knobs.
///
/// `#[non_exhaustive]` on purpose: fields get added, and a struct literal at a
/// call site would then be a breaking change. Build one with
/// [`TranscribeOptions::default`] plus the `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TranscribeOptions {
    /// How words are grouped into subtitle segments. Defaults to
    /// [`Limits::default`] — the policy the regression baseline pins.
    pub segments: Limits,
}

impl TranscribeOptions {
    /// Replace the whole segmentation policy.
    #[must_use]
    pub fn with_segments(mut self, limits: Limits) -> Self {
        self.segments = limits;
        self
    }

    /// Silence in seconds that closes a segment.
    #[must_use]
    pub fn with_segment_gap(mut self, seconds: f32) -> Self {
        self.segments.gap_seconds = seconds;
        self
    }

    /// Soft cap on segment text length in characters.
    #[must_use]
    pub fn with_segment_max_chars(mut self, chars: usize) -> Self {
        self.segments.max_chars = chars;
        self
    }

    /// Hard cap on segment duration in seconds.
    #[must_use]
    pub fn with_segment_max_seconds(mut self, seconds: f32) -> Self {
        self.segments.max_seconds = seconds;
        self
    }
}

/// Everything that only exists on the wgpu path: the adapter, the uploaded
/// weights and every buffer derived from them.
///
/// Kept in one value so a CPU-only process allocates none of it — no adapter,
/// no 1.2 GB of VRAM, and no requirement that the machine has a GPU at all.
#[doc(hidden)]
pub struct GpuState {
    pub enc: Encoder,
    pub decode: GpuTdtDecoder,
    pub sub: SubGpu,
}

/// A loaded model: one instance transcribes any number of files.
pub struct AsrInference {
    /// `None` on the CPU backend.
    #[doc(hidden)]
    pub gpu: Option<GpuState>,
    /// CPU TDT decoder. Always present: the GPU path uses it only for the
    /// `BARK3_CPU_DEC=1` comparison.
    #[doc(hidden)]
    pub tdt: TdtDecoder,
    #[doc(hidden)]
    pub sub_w: SubsamplingWeights,
    /// `Some` when the CPU encoder is loaded (CPU backend, or a diagnostic that
    /// compares both).
    #[doc(hidden)]
    pub cpu: Option<CpuEncoder>,
    #[doc(hidden)]
    pub mel_fe: MelFrontend,
    /// The backend that actually runs, after [`Backend::Auto`] resolved.
    backend: Backend,
    /// What `model_dir` turned out to hold, read from its header at load time.
    info: crate::weights_io::ModelInfo,
    model_dir: PathBuf,
}

impl AsrInference {
    /// Load weights from an official checkpoint directory and build `backend`.
    ///
    /// The directory may hold either layout the releases ship: a
    /// `model.safetensors` (full precision) or a `model.bark` container (the
    /// compressed releases, whose encoder linears stay 4-bit). Nothing has to be
    /// converted first, and nothing is guessed: the header is read, the
    /// architecture is checked against this build, and only then are weights
    /// uploaded.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(model_dir: impl AsRef<Path>, backend: Backend) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let src = crate::weights::open(&model_dir)?;
        let mut asr = pollster::block_on(Self::load_source(src, backend))?;
        // Diagnostics reopen the directory to build the second encoder.
        asr.model_dir = model_dir;
        Ok(asr)
    }

    /// Async loader over an explicit source: the entry point for the wasm build,
    /// where the weights are fetched into memory instead of read from a path.
    pub async fn load_source(
        src: Box<dyn crate::weights::WeightSource>,
        backend: Backend,
    ) -> Result<Self> {
        let model_dir = PathBuf::new();
        let info = src.info().clone();
        info.check_architecture()?;
        crate::browser::log("[load] source open");
        eprintln!("[model] {}", info.summary());
        let started = crate::clock::now();

        let get = |n: &str| src.read(n);
        let tdt = TdtDecoder::from_source(&get)?;
        crate::browser::log("[load] tdt weights read");
        let sub_w = SubsamplingWeights::from_source(&get)?;
        crate::browser::log("[load] subsampling weights read");

        let gpu = match backend {
            Backend::Cpu => None,
            Backend::Gpu => Some(Self::load_gpu(src.as_ref(), &tdt, &sub_w).await?),
            Backend::Auto => match Self::load_gpu(src.as_ref(), &tdt, &sub_w).await {
                Ok(gpu) => Some(gpu),
                Err(err) => {
                    eprintln!("[gpu] unavailable ({err:#}); falling back to the CPU backend");
                    None
                }
            },
        };
        let cpu = if gpu.is_some() {
            None
        } else {
            Some(Self::load_cpu(src.as_ref())?)
        };
        let backend = if gpu.is_some() {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        #[cfg(target_arch = "wasm32")]
        if gpu.is_none() {
            eprintln!(
                "[gpu] no WebGPU adapter; running the CPU backend single-threaded                  (slow: the browser has no threads to give it)"
            );
        }
        eprintln!(
            "[load] {} backend in {:.2}s",
            backend.tag(),
            started.elapsed().as_secs_f32()
        );

        crate::browser::log("[load] inference assembled");
        Ok(Self {
            gpu,
            tdt,
            sub_w,
            cpu,
            mel_fe: MelFrontend::new(mel_filters()),
            backend,
            info,
            model_dir,
        })
    }

    async fn load_gpu(
        src: &dyn WeightSource,
        tdt: &TdtDecoder,
        sub_w: &SubsamplingWeights,
    ) -> Result<GpuState> {
        // `BARK3_DEVICE=<selector>` pins the adapter (e.g. `intel`, `dx12:1`,
        // `vulkan:0`); without it Auto takes the best device it can open. The
        // browser has no environment, so there it stays Auto.
        let selector = match std::env::var("BARK3_DEVICE") {
            Ok(v) => DeviceSelector::parse(&v)?,
            Err(_) => DeviceSelector::Auto,
        };
        let gpu = Gpu::new_with(selector).await?;
        eprintln!("[gpu] {}", gpu.describe());
        let enc = Encoder::load_with(gpu, src)?;
        crate::browser::log("[load] encoder built");
        let decode = GpuTdtDecoder::new(&enc.gpu, tdt)?;
        crate::browser::log("[load] tdt decoder built");
        let sub = SubGpu::new(&enc.gpu, sub_w)?;
        crate::browser::log("[load] subsampling built");
        Ok(GpuState { enc, decode, sub })
    }

    fn load_cpu(src: &dyn WeightSource) -> Result<CpuEncoder> {
        let started = crate::clock::now();
        let cpu = CpuEncoder::load_with(src)?;
        eprintln!(
            "[load] cpu encoder in {:.2}s",
            started.elapsed().as_secs_f32()
        );
        Ok(cpu)
    }

    /// Transcribes audio that a VAD split into speech runs, and returns one
    /// timeline for the whole recording.
    ///
    /// `cuts` are absolute times in seconds — typically the *midpoints of the
    /// silences* between speech runs. Each chunk is decoded with `PAD` seconds of
    /// context on both sides so a word that starts on the boundary is not clipped,
    /// then a word is kept by exactly one chunk: whichever span contains its
    /// midpoint. Word times are shifted to the recording's timeline, so the result
    /// has the same shape (and the same JSON) as a single-pass transcription.
    ///
    /// This is what the browser build feeds: one call per recording, after the
    /// VAD has decided where the silences are.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn transcribe_chunked(
        &self,
        samples: &[f32],
        cuts: &[f32],
        options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let Some(spans) = self.chunk_spans(samples, cuts) else {
            return self.transcribe_samples(samples, options);
        };
        let mut acc = ChunkAcc::new(samples.len());
        for span in spans {
            let chunk = self.transcribe_samples(&samples[span.a..span.b], options)?;
            acc.add(chunk, &span);
        }
        Ok(acc.finish(options))
    }

    /// The browser's form of [`Self::transcribe_chunked`]: each chunk comes back
    /// through a buffer map, so the loop has to be awaited.
    pub async fn transcribe_chunked_async(
        &self,
        samples: &[f32],
        cuts: &[f32],
        options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let Some(spans) = self.chunk_spans(samples, cuts) else {
            return self.transcribe_samples_async(samples, options).await;
        };
        let mut acc = ChunkAcc::new(samples.len());
        for span in spans {
            let chunk = self
                .transcribe_samples_async(&samples[span.a..span.b], options)
                .await?;
            acc.add(chunk, &span);
        }
        Ok(acc.finish(options))
    }

    /// Cut the recording into the spans that get decoded one at a time.
    ///
    /// `None` means "no usable cut": the caller should decode the recording in
    /// one pass instead.
    fn chunk_spans(&self, samples: &[f32], cuts: &[f32]) -> Option<Vec<Span>> {
        /// Context on each side of a cut, so boundary words decode in full.
        const PAD: f32 = 0.20;
        let duration = samples.len() as f32 / SAMPLE_RATE as f32;
        let mut edges: Vec<f32> = cuts
            .iter()
            .copied()
            .filter(|c| *c > 0.0 && *c < duration)
            .collect();
        edges.sort_by(f32::total_cmp);
        edges.dedup();
        if edges.is_empty() {
            return None;
        }
        let mut bounds = Vec::with_capacity(edges.len() + 1);
        let mut start = 0.0f32;
        for edge in edges {
            bounds.push((start, edge));
            start = edge;
        }
        bounds.push((start, duration));
        let mut spans = Vec::with_capacity(bounds.len());
        for (from, to) in bounds {
            let lo = (from - PAD).max(0.0);
            let hi = (to + PAD).min(duration);
            let a = (lo * SAMPLE_RATE as f32) as usize;
            let b = ((hi * SAMPLE_RATE as f32) as usize).min(samples.len());
            if b <= a {
                continue;
            }
            spans.push(Span { from, to, lo, a, b });
        }
        Some(spans)
    }

    /// Build the CPU encoder in addition to whatever is already loaded, so the
    /// two backends can be compared. Diagnostic binaries only.
    #[doc(hidden)]
    pub fn load_cpu_encoder(&mut self) -> Result<()> {
        if self.cpu.is_some() {
            return Ok(());
        }
        self.cpu = Some(Self::load_cpu(
            crate::weights::open(&self.model_dir)?.as_ref(),
        )?);
        Ok(())
    }

    /// The backend actually in use, after [`Backend::Auto`] resolved.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Which release this instance loaded, and in which weight layout.
    ///
    /// Read from the checkpoint header during [`Self::load`], which also refuses
    /// an architecture this build was not compiled for.
    #[must_use]
    pub fn model_info(&self) -> &crate::weights_io::ModelInfo {
        &self.info
    }

    /// Human-readable description of the compute device.
    #[must_use]
    pub fn device_description(&self) -> String {
        self.gpu
            .as_ref()
            .map_or_else(|| "cpu (no adapter)".to_string(), |g| g.enc.gpu.describe())
    }

    /// The wgpu backend, or an error naming the reason it is absent.
    pub fn gpu(&self) -> Result<&GpuState> {
        self.gpu
            .as_ref()
            .context("this instance is running the CPU backend; load with Backend::Gpu")
    }

    /// Transcribe a 16 kHz mono WAV file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn transcribe(
        &self,
        path: impl AsRef<Path>,
        options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        let path = path.as_ref();
        let wave = read_wav(path)?;
        self.transcribe_samples(&wave, options)
            .with_context(|| format!("transcribing {}", path.display()))
    }

    /// Transcribe in-memory 16 kHz mono samples.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        crate::transcription::run(self, samples, None, &options)
    }

    /// The browser's form: readbacks resolve on the JS event loop, so the whole
    /// transcription has to be awaited.
    pub async fn transcribe_samples_async(
        &self,
        samples: &[f32],
        options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        crate::transcription::run_async(self, samples, None, &options).await
    }
}

/// The bundled mel filterbank, `[128, 257]` f32.
fn mel_filters() -> Vec<f32> {
    MEL_FILTERS
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One chunk of the VAD-driven split: `[a, b)` of the sample buffer, decoded with
/// `lo` seconds of lead-in context and owned from `from` to `to`.
struct Span {
    from: f32,
    to: f32,
    lo: f32,
    a: usize,
    b: usize,
}

/// Accumulates per-chunk results into one recording timeline.
struct ChunkAcc {
    words: Vec<crate::transcript::Word>,
    tokens: Vec<crate::tdt::Token>,
    texts: Vec<String>,
    duration: f32,
}

impl ChunkAcc {
    fn new(sample_count: usize) -> Self {
        Self {
            words: Vec::new(),
            tokens: Vec::new(),
            texts: Vec::new(),
            duration: sample_count as f32 / SAMPLE_RATE as f32,
        }
    }

    /// Keep the words and tokens this chunk owns: a word belongs to whichever
    /// span contains its midpoint, so a word straddling a cut appears once.
    fn add(&mut self, chunk: TranscribeResult, span: &Span) {
        let frame_of = |seconds: f32| (seconds / crate::transcript::FRAME_SECONDS).round() as u32;
        if !chunk.text.is_empty() {
            self.texts.push(chunk.text);
        }
        let frame_base = frame_of(span.lo);
        for token in chunk.tokens {
            let frame = token.frame + frame_base;
            if frame >= frame_of(span.from) && frame < frame_of(span.to) {
                self.tokens.push(crate::tdt::Token { frame, ..token });
            }
        }
        for mut word in chunk.words {
            word.start += span.lo;
            word.end = (word.end + span.lo).min(self.duration);
            let middle = 0.5 * (word.start + word.end);
            if middle >= span.from && middle < span.to {
                self.words.push(word);
            }
        }
    }

    fn finish(mut self, options: TranscribeOptions) -> TranscribeResult {
        self.words.sort_by(|a, b| a.start.total_cmp(&b.start));
        TranscribeResult {
            text: self.texts.join(" "),
            segments: crate::segment::words_to_segments(
                &self.words,
                self.duration,
                &options.segments,
            ),
            words: self.words,
            tokens: self.tokens,
            duration: self.duration,
        }
    }
}
