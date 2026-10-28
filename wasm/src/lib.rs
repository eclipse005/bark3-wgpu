//! Browser bindings: the same crate, compiled to `wasm32-unknown-unknown`.
//!
//! JS side:
//!
//! ```js
//! import init, { BarkAsr } from "./bark3_wasm.js";
//!
//! await init();
//! const bytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
//! const asr = await BarkAsr.create(bytes, "Underdog-Bark-3.0-Plus", "auto");
//!
//! // 16 kHz mono f32 samples, and the silence midpoints a VAD produced:
//! const json = JSON.parse(asr.transcribeChunked(samples, cuts));
//! // json.words === [{ text, start, end }, ...] — the same shape the native
//! // binary prints, so sentence splitting and translation stay unchanged.
//! ```
//!
//! One recording, one call: the VAD pairs (silero or otherwise) supply the cut
//! times, the Rust side decodes each speech run with 0.2 s of context and returns
//! a single timeline. Everything below the bindings — mel, encoder, TDT, word and
//! segment assembly — is the library the CLI uses, so the JSON matches what
//! `bark3 transcribe` prints for the same audio.

use bark3_wgpu::{AsrInference, Backend, TranscribeOptions};
use wasm_bindgen::prelude::*;

/// A loaded model. Created once per page, reused for every recording.
#[wasm_bindgen]
pub struct BarkAsr {
    inner: AsrInference,
}

#[wasm_bindgen]
impl BarkAsr {
    /// Loads a checkpoint from its bytes.
    ///
    /// `model` is the whole official file — `model.bark` for a compressed release
    /// or `model.safetensors` for the full-precision one — as fetched by the page.
    /// `backend` is `"auto"` (default), `"gpu"` or `"cpu"`.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(
        model: Vec<u8>,
        release: Option<String>,
        backend: Option<String>,
    ) -> Result<BarkAsr, JsError> {
        console_error_panic_hook::set_once();
        let backend = match backend.as_deref().unwrap_or("auto") {
            "cpu" => Backend::Cpu,
            "gpu" => Backend::Gpu,
            _ => Backend::Auto,
        };
        let bytes = std::sync::Arc::new(model);
        // A compressed release is worth caching (its load is dominated by ANS /
        // LZMA decoding); a safetensors checkpoint is already fast and would
        // only produce a gigabyte of cache.
        // Only the GPU path reads the 4-bit codes as codes. The CPU encoder
        // widens every weight to f32, so recording it there would cache ~2.5 GB
        // (and run the wasm heap out of memory) instead of the ~400 MB the GPU
        // path needs.
        let cacheable =
            backend != Backend::Cpu && bark3_wgpu::weights::looks_like_container(bytes.as_slice());
        let key = bark3_wgpu::weights_cache::fingerprint(&bytes, release.as_deref());
        let mut recorder = None;
        let source: Box<dyn bark3_wgpu::weights::WeightSource> =
            if cacheable && let Ok(Some(cached)) = bark3_wgpu::weights_cache::load(key).await {
                bark3_wgpu::browser::log(&format!("[wasm] weights from cache {key:016x}"));
                cached
            } else {
                let fresh = match bark3_wgpu::weights::open_bytes(bytes.clone(), release) {
                    Ok(src) => src,
                    Err(e) => return Err(JsError::new(&format!("{e:#}"))),
                };
                if cacheable {
                    let (recorded, rec) = bark3_wgpu::weights_cache::record(fresh);
                    recorder = Some(rec);
                    recorded
                } else {
                    fresh
                }
            };
        let inner = AsrInference::load_source(source, backend)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        if let Some(rec) = recorder {
            // Written in the background: the page can start transcribing while
            // the decoded tensors go to disk for the next load.
            wasm_bindgen_futures::spawn_local(async move {
                match rec.persist(key).await {
                    Ok(()) => bark3_wgpu::browser::log("[wasm] weights cached for the next load"),
                    Err(e) => {
                        bark3_wgpu::browser::log(&format!("[wasm] cache write failed: {e:#}"));
                    }
                }
            });
        }
        bark3_wgpu::browser::log("[wasm] create: inference ready");
        if inner.backend() != Backend::Cpu {
            // Dawn compiles each pipeline at its first dispatch; doing that here
            // keeps it off the first real transcription (see `warmup_async`).
            let t = bark3_wgpu::clock::now();
            match inner.warmup_async().await {
                Ok(()) => bark3_wgpu::browser::log(&format!(
                    "[wasm] warmup {:.2}s",
                    t.elapsed().as_secs_f32()
                )),
                Err(e) => bark3_wgpu::browser::log(&format!("[wasm] warmup failed: {e:#}")),
            }
        }
        let out = BarkAsr { inner };
        bark3_wgpu::browser::log("[wasm] create: returning");
        Ok(out)
    }

    /// One JSON document: `{ text, duration, segments, words }`, the same the CLI
    /// prints. `samples` are 16 kHz mono f32.
    pub async fn transcribe(&self, samples: Vec<f32>) -> Result<String, JsError> {
        self.inner
            .transcribe_samples_async(&samples, TranscribeOptions::default())
            .await
            .map(|r| r.to_json())
            .map_err(|e| JsError::new(&format!("{e:#}")))
    }

    /// Transcribes the whole recording with VAD-driven cuts.
    ///
    /// `cuts` are absolute times in seconds; the natural choice is the midpoint of
    /// each silence between two speech runs. Word times come back on the
    /// recording's timeline, in one JSON document.
    #[wasm_bindgen(js_name = transcribeChunked)]
    pub async fn transcribe_chunked(
        &self,
        samples: Vec<f32>,
        cuts: Vec<f32>,
    ) -> Result<String, JsError> {
        self.inner
            .transcribe_chunked_async(&samples, &cuts, TranscribeOptions::default())
            .await
            .map(|r| r.to_json())
            .map_err(|e| JsError::new(&format!("{e:#}")))
    }

    /// Releases the GPU device — every buffer it owns goes back to the driver now.
    ///
    /// Call it when the page stops transcribing: WebGPU only returns a dropped
    /// buffer once its JavaScript wrapper is collected, so without this the card
    /// stays occupied until the tab closes. The instance cannot transcribe
    /// afterwards; create a new one.
    pub fn close(&self) {
        self.inner.release_gpu();
        bark3_wgpu::browser::log("[wasm] device destroyed");
    }

    /// What was loaded: release name, weight layout, parameter count, architecture.
    #[wasm_bindgen(js_name = modelInfo)]
    pub fn model_info(&self) -> String {
        let info = self.inner.model_info();
        serde_json::json!({
            "release": info.release,
            "layout": info.layout.tag(),
            "weights": info.weights,
            "layers": info.layers,
            "hidden": info.hidden,
            "ffn": info.ffn,
            "vocab": info.vocab,
        })
        .to_string()
    }

    /// `"gpu"` or `"cpu"`, after `auto` resolved.
    pub fn backend(&self) -> String {
        self.inner.backend().tag().to_string()
    }

    /// The device that runs the encoder (adapter name, or `cpu`).
    pub fn device(&self) -> String {
        self.inner.device_description()
    }

    /// Largest audio this instance accepts, in seconds. The encoder pads to whole
    /// frames and its positional embedding reaches 5000 frames (400 s); the page
    /// should cut long recordings with the VAD instead of relying on this.
    #[wasm_bindgen(js_name = maxSeconds)]
    pub fn max_seconds() -> f32 {
        400.0
    }
}
