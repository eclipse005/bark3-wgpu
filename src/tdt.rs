//! TDT greedy decoding on CPU (f32) and transcript assembly.
//!
//! The decode loop mirrors transformers' ParakeetTDTGenerationMixin: each step
//! predicts a token and a duration. Duration advances the frame on every step;
//! blank additionally forces at least one frame. Worker scheduling and its raw
//! pointer boundary are isolated in `pool`.

mod pool;

use rayon::prelude::*;

use crate::cfg;
use crate::raw::RawTensor;
use pool::LstmPool;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct TdtDecoder {
    pub embed: Vec<f32>,         // (VOCAB, 640) f32
    pub(crate) w_ih: Vec<u16>,   // (2, 2560, 640) layer-major, f16
    pub(crate) w_hh: Vec<u16>,   // (2, 2560, 640) f16
    pub(crate) b_ih: Vec<f32>,   // (2, 2560)
    pub(crate) b_hh: Vec<f32>,   // (2, 2560)
    pub(crate) proj_w: Vec<u16>, // (640, 640) f16
    pub(crate) proj_b: Vec<f32>, // (640)
    pub(crate) head_w: Vec<u16>, // (8198, 640) f16
    pub(crate) head_b: Vec<f32>, // (8198)
}

/// One emitted token and where it lives on the decode timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    /// Frame index at emission (80 ms units).
    pub frame: u32,
    /// Frames the decoder advanced after emitting this token (the TDT duration
    /// class, 0..=4). `frame + dur` is where the next step starts, so a word
    /// ends at its last token's `frame + dur` rather than at the next word.
    pub dur: u32,
}

const DH: usize = cfg::DEC_HIDDEN;
const GS: usize = 4 * DH;

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn pack16(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot16_avx(w: *const u16, x: *const f32, n: usize) -> f32 {
    // SAFETY: the caller provides at least `n` readable f16/f32 elements. The
    // loop bounds below permit 8/16-wide reads only while they remain in range.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let h0 = _mm_loadu_si128(w.add(i) as *const __m128i);
            let h1 = _mm_loadu_si128(w.add(i + 8) as *const __m128i);
            acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(h0), _mm256_loadu_ps(x.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(h1), _mm256_loadu_ps(x.add(i + 8)), acc1);
            i += 16;
        }
        let mut acc = _mm256_add_ps(acc0, acc1);
        while i + 8 <= n {
            let h = _mm_loadu_si128(w.add(i) as *const __m128i);
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(x.add(i)), acc);
            i += 8;
        }
        let mut tmp = [0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = 0f32;
        for value in tmp {
            sum += value;
        }
        while i < n {
            sum += f16_to_f32(*w.add(i)) * *x.add(i);
            i += 1;
        }
        sum
    }
}

#[inline]
fn dot16(w: &[u16], x: &[f32], simd: bool) -> f32 {
    debug_assert!(x.len() >= w.len());
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `x` has at least `w.len()` elements and AVX2/F16C/FMA support
        // was checked by `TdtDecoder::simd_ok`.
        unsafe {
            return dot16_avx(w.as_ptr(), x.as_ptr(), w.len());
        }
    }
    let mut sum = 0f32;
    for i in 0..w.len() {
        sum += f16_to_f32(w[i]) * x[i];
    }
    sum
}

/// One tensor, then converted: the two accessors keep `from_source` free of
/// nested closures.
fn f32_of(get: &dyn Fn(&str) -> anyhow::Result<RawTensor>, name: &str) -> anyhow::Result<Vec<f32>> {
    Ok(get(name)?.to_f32())
}

fn joined(
    get: &dyn Fn(&str) -> anyhow::Result<RawTensor>,
    a: &str,
    b: &str,
) -> anyhow::Result<Vec<f32>> {
    let (mut joined, second) = (f32_of(get, a)?, f32_of(get, b)?);
    joined.extend_from_slice(&second);
    Ok(joined)
}

impl TdtDecoder {
    pub fn from_tensors(weights: &std::collections::HashMap<String, RawTensor>) -> Self {
        Self::from_source(&|name| {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))
        })
        .expect("weight map is complete")
    }

    /// Builds the decoder one tensor at a time, so a streaming reader never has
    /// to hold the whole checkpoint.
    pub fn from_source(get: &dyn Fn(&str) -> anyhow::Result<RawTensor>) -> anyhow::Result<Self> {
        Ok(Self {
            embed: f32_of(get, "decoder.embedding.weight")?,
            w_ih: pack16(&joined(
                get,
                "decoder.lstm.weight_ih_l0",
                "decoder.lstm.weight_ih_l1",
            )?),
            w_hh: pack16(&joined(
                get,
                "decoder.lstm.weight_hh_l0",
                "decoder.lstm.weight_hh_l1",
            )?),
            b_ih: joined(get, "decoder.lstm.bias_ih_l0", "decoder.lstm.bias_ih_l1")?,
            b_hh: joined(get, "decoder.lstm.bias_hh_l0", "decoder.lstm.bias_hh_l1")?,
            proj_w: pack16(&f32_of(get, "decoder.decoder_projector.weight")?),
            proj_b: f32_of(get, "decoder.decoder_projector.bias")?,
            head_w: pack16(&f32_of(get, "joint.head.weight")?),
            head_b: f32_of(get, "joint.head.bias")?,
        })
    }

    #[inline]
    fn simd_ok() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Apply one two-layer LSTM step to an input-token embedding.
    pub fn lstm_step(&self, x: &[f32], h: &mut [Vec<f32>], c: &mut [Vec<f32>]) {
        let simd = Self::simd_ok();
        for layer in 0..2 {
            let wih = &self.w_ih[layer * GS * DH..(layer + 1) * GS * DH];
            let whh = &self.w_hh[layer * GS * DH..(layer + 1) * GS * DH];
            let bih = &self.b_ih[layer * GS..(layer + 1) * GS];
            let bhh = &self.b_hh[layer * GS..(layer + 1) * GS];
            let xin: &[f32] = if layer == 0 { x } else { &h[0] };
            let hin = &h[layer];
            let mut gates = vec![0f32; GS];
            gates.par_iter_mut().enumerate().for_each(|(row, out)| {
                let wi = &wih[row * DH..(row + 1) * DH];
                let wh = &whh[row * DH..(row + 1) * DH];
                *out = bih[row] + bhh[row] + dot16(wi, xin, simd) + dot16(wh, hin, simd);
            });
            for channel in 0..DH {
                let input = sigmoid(gates[channel]);
                let forget = sigmoid(gates[DH + channel]);
                let candidate = gates[2 * DH + channel].tanh();
                c[layer][channel] = forget * c[layer][channel] + input * candidate;
                let output = sigmoid(gates[3 * DH + channel]);
                h[layer][channel] = output * c[layer][channel].tanh();
            }
        }
    }

    /// Joint argmax over token and duration rows, parallel over row chunks.
    fn joint(
        &self,
        enc_frame: &[f32],
        dec: &[f32],
        scratch: &mut [f32],
        frame: usize,
    ) -> (u32, u32) {
        for i in 0..DH {
            scratch[i] = (enc_frame[i] + dec[i]).max(0.0);
        }
        let rows = cfg::VOCAB + cfg::N_DUR;
        let simd = Self::simd_ok();
        const CHUNKS: usize = 64;
        let per = rows.div_ceil(CHUNKS);
        let partials: Vec<(f32, u32, f32, u32)> = (0..CHUNKS)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * per;
                let end = (start + per).min(rows);
                let mut best_token = (f32::NEG_INFINITY, start as u32);
                let mut best_duration = (f32::NEG_INFINITY, start as u32);
                for row_index in start..end {
                    let row = &self.head_w[row_index * DH..(row_index + 1) * DH];
                    let score = self.head_b[row_index] + dot16(row, scratch, simd);
                    if row_index < cfg::VOCAB {
                        if score > best_token.0 {
                            best_token = (score, row_index as u32);
                        }
                    } else if score > best_duration.0 {
                        best_duration = (score, row_index as u32);
                    }
                }
                (best_token.0, best_token.1, best_duration.0, best_duration.1)
            })
            .collect();
        let mut best_token = (f32::NEG_INFINITY, 0u32);
        let mut best_duration = (f32::NEG_INFINITY, 0u32);
        for (token_score, token_id, duration_score, duration_id) in partials {
            if token_score > best_token.0 {
                best_token = (token_score, token_id);
            }
            if duration_score > best_duration.0 {
                best_duration = (duration_score, duration_id);
            }
        }
        if std::env::var("BARK3_DECDUMP").is_ok() {
            let dec_sum: f32 = dec.iter().sum();
            let enc_sum: f32 = enc_frame.iter().sum();
            eprintln!("[dec] frame={frame} decsum={dec_sum:.6} encsum={enc_sum:.6}");
        }
        if std::env::var("BARK3_JOINTDBG").is_ok() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/vocab.json");
            let vocab: std::collections::HashMap<String, usize> =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap_or_default();
            let name = |row: u32| {
                vocab
                    .iter()
                    .find(|(_, index)| **index as u32 == row)
                    .map(|(s, _)| s.clone())
                    .unwrap_or_default()
            };
            eprintln!(
                "[joint] frame={frame} tok={} (named {}) dur={}",
                best_token.1,
                name(best_token.1),
                best_duration.1
            );
        }
        (best_token.1, best_duration.1 - cfg::VOCAB as u32)
    }

    /// Deterministic initial state for diagnostics and alternative backends.
    pub fn init_state(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h = vec![vec![0f32; DH]; 2];
        let mut c = vec![vec![0f32; DH]; 2];
        let embedding = &self.embed[cfg::BLANK as usize * DH..cfg::BLANK as usize * DH + DH];
        self.lstm_step(embedding, &mut h, &mut c);
        let dec = self.project(&h[1]);
        let mut flat_h = Vec::with_capacity(2 * DH);
        let mut flat_c = Vec::with_capacity(2 * DH);
        for layer in 0..2 {
            flat_h.extend_from_slice(&h[layer]);
            flat_c.extend_from_slice(&c[layer]);
        }
        (flat_h, flat_c, dec)
    }

    /// Greedy decode over pooler output (T x 640).
    ///
    /// The decoder is a spin-pool over the LSTM gates and the joint head, which
    /// needs scoped threads; the browser build uses the GPU decoder instead (and
    /// refuses to load without an adapter, see `AsrInference::load_source`).
    #[cfg(target_arch = "wasm32")]
    pub fn decode(&self, _pooler: &[f32], _t_len: usize) -> Vec<Token> {
        panic!("the CPU TDT decoder needs threads; the browser build decodes on the GPU")
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode(&self, pooler: &[f32], t_len: usize) -> Vec<Token> {
        if std::env::var("BARK3_RAYON_DEC").is_ok() {
            return self.decode_rayon(pooler, t_len);
        }
        let worker_count = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(8)
            .saturating_sub(2)
            .clamp(4, 16);
        let pool = LstmPool::new(self, worker_count);
        std::thread::scope(|scope| {
            for worker_index in 0..worker_count {
                let pool = &pool;
                scope.spawn(move || pool.worker(worker_index));
            }
            let result = self.decode_with_pool(&pool, pooler, t_len);
            pool.stop();
            result
        })
    }

    fn lstm_pooled(
        &self,
        pool: &LstmPool<'_>,
        x: &[f32],
        h: &mut [Vec<f32>],
        c: &mut [Vec<f32>],
        gates: &mut [f32],
    ) {
        assert_eq!(gates.len(), 2 * GS);
        for layer in 0..2 {
            let xin: Vec<f32> = if layer == 0 { x.to_vec() } else { h[0].clone() };
            let layer_gates = &mut gates[layer * GS..(layer + 1) * GS];
            pool.gates(layer, &xin, &h[layer], layer_gates);
            pool.update(layer_gates, &mut h[layer], &mut c[layer]);
        }
    }

    fn joint_pooled(
        &self,
        pool: &LstmPool<'_>,
        enc: &[f32],
        dec: &[f32],
        scratch: &mut [f32],
    ) -> (u32, u32) {
        for i in 0..DH {
            scratch[i] = (enc[i] + dec[i]).max(0.0);
        }
        pool.joint(scratch)
    }

    fn project_pooled(&self, pool: &LstmPool<'_>, x: &[f32], out: &mut [f32]) {
        pool.project(x, out);
    }

    fn decode_rayon(&self, pooler: &[f32], t_len: usize) -> Vec<Token> {
        let mut h = vec![vec![0f32; DH]; 2];
        let mut c = vec![vec![0f32; DH]; 2];
        let mut scratch = vec![0f32; DH];
        let mut tokens = Vec::new();
        let mut frame = 0usize;
        let mut guard = 0usize;
        let max_steps = t_len.saturating_mul(cfg::MAX_SYMBOLS as usize + 4) + 64;
        let blank_embedding =
            self.embed[cfg::BLANK as usize * DH..cfg::BLANK as usize * DH + DH].to_vec();
        self.lstm_step(&blank_embedding, &mut h, &mut c);
        let mut dec = self.project(&h[1]);
        loop {
            if frame >= t_len {
                break;
            }
            guard += 1;
            if guard > max_steps {
                break;
            }
            let enc_frame = &pooler[frame * DH..(frame + 1) * DH];
            let (token, duration) = self.joint(enc_frame, &dec, &mut scratch, frame);
            if token == cfg::BLANK {
                frame += duration.max(1) as usize;
            } else {
                tokens.push(Token {
                    id: token,
                    frame: frame as u32,
                    dur: duration,
                });
                self.lstm_step(
                    &self.embed[token as usize * DH..token as usize * DH + DH],
                    &mut h,
                    &mut c,
                );
                dec = self.project(&h[1]);
                frame += duration as usize;
            }
        }
        tokens
    }

    fn decode_with_pool(&self, pool: &LstmPool<'_>, pooler: &[f32], t_len: usize) -> Vec<Token> {
        let mut h = vec![vec![0f32; DH]; 2];
        let mut c = vec![vec![0f32; DH]; 2];
        let mut scratch = vec![0f32; DH];
        let mut gates = vec![0f32; 2 * GS];
        let mut tokens = Vec::new();
        let mut frame = 0usize;
        let mut guard = 0usize;
        let max_steps = t_len.saturating_mul(cfg::MAX_SYMBOLS as usize + 4) + 64;
        let blank_embedding =
            self.embed[cfg::BLANK as usize * DH..cfg::BLANK as usize * DH + DH].to_vec();
        self.lstm_pooled(pool, &blank_embedding, &mut h, &mut c, &mut gates);
        let mut dec = vec![0f32; DH];
        self.project_pooled(pool, &h[1], &mut dec);
        loop {
            if frame >= t_len {
                eprintln!(
                    "[tdt] frame exit: frame={frame} t_len={t_len} steps={guard} tokens={}",
                    tokens.len()
                );
                break;
            }
            guard += 1;
            if guard > max_steps {
                eprintln!("[tdt] guard exit: frame={frame} steps={guard}");
                break;
            }
            let enc_frame = &pooler[frame * DH..(frame + 1) * DH];
            let (token, duration) = if std::env::var("BARK3_POOL_JOINT").is_ok() {
                self.joint_pooled(pool, enc_frame, &dec, &mut scratch)
            } else {
                self.joint(enc_frame, &dec, &mut scratch, frame)
            };
            if token == cfg::BLANK {
                frame += duration.max(1) as usize;
            } else {
                tokens.push(Token {
                    id: token,
                    frame: frame as u32,
                    dur: duration,
                });
                self.lstm_pooled(
                    pool,
                    &self.embed[token as usize * DH..token as usize * DH + DH],
                    &mut h,
                    &mut c,
                    &mut gates,
                );
                self.project_pooled(pool, &h[1], &mut dec);
                frame += duration as usize;
            }
        }
        tokens
    }

    /// Microbenchmark the three per-step primitives (diagnostics only).
    pub fn bench(&self, reps: usize) {
        use std::time::Instant;
        let enc = vec![0.01f32; DH];
        let dec = vec![0.02f32; DH];
        let mut scratch = vec![0f32; DH];
        let mut sink = 0u64;
        let started = Instant::now();
        for _ in 0..reps {
            let (token, duration) = self.joint(&enc, &dec, &mut scratch, 0);
            sink = sink.wrapping_add(token as u64 + duration as u64);
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "[bench] joint   {:8.1} us/call",
            elapsed * 1e6 / reps as f64
        );
        let mut h = vec![vec![0f32; DH]; 2];
        let mut c = vec![vec![0f32; DH]; 2];
        let x = vec![0.01f32; DH];
        let started = Instant::now();
        for _ in 0..reps {
            self.lstm_step(&x, &mut h, &mut c);
            sink = sink.wrapping_add(h[1][0].to_bits() as u64);
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "[bench] lstm    {:8.1} us/call",
            elapsed * 1e6 / reps as f64
        );
        let started = Instant::now();
        for _ in 0..reps {
            let dec = self.project(&x);
            sink = sink.wrapping_add(dec[0].to_bits() as u64);
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "[bench] project {:8.1} us/call",
            elapsed * 1e6 / reps as f64
        );
        println!("[bench] sink={sink} (ignore)");
    }

    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let simd = Self::simd_ok();
        let mut out = vec![0f32; DH];
        out.par_iter_mut().enumerate().for_each(|(row_index, y)| {
            let row = &self.proj_w[row_index * DH..(row_index + 1) * DH];
            *y = self.proj_b[row_index] + dot16(row, x, simd);
        });
        out
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
