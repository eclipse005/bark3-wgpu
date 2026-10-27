//! CPU conv subsampling (exact torch semantics): Conv2d s2 -> ReLU ->
//! dw Conv2d s2 -> pw Conv2d -> ReLU -> dw s2 -> pw -> ReLU, masked lengths,
//! then flatten (256 x T x 16) -> rows (T, 4096) for the GPU linear.
//! The 4096->1024 projection runs on GPU as a GEMM.

use rayon::prelude::*;

use crate::cfg;
use crate::raw::RawTensor;

fn conv_len(l: usize) -> usize {
    (l + 2 - 3) / 2 + 1
}

/// dst += a * src, 8 lanes at a time.  Separate mul+add (not fma) keeps each
/// output's rounding identical to the scalar accumulation it replaces.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpy_avx2(dst: &mut [f32], src: &[f32], a: f32) {
    use std::arch::x86_64::*;
    assert!(src.len() >= dst.len());
    let n = dst.len();
    // SAFETY: AVX2 support is checked by the caller. The loop reads/writes only
    // full 8-lane chunks inside `dst`; `src.len() >= dst.len()` covers the loads.
    unsafe {
        let av = _mm256_set1_ps(a);
        let mut i = 0usize;
        while i + 8 <= n {
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i),
                _mm256_add_ps(d, _mm256_mul_ps(av, s)),
            );
            i += 8;
        }
        while i < n {
            dst[i] += a * src[i];
            i += 1;
        }
    }
}

#[inline]
fn axpy(dst: &mut [f32], src: &[f32], a: f32, simd: bool) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        unsafe { axpy_avx2(dst, src, a) };
        return;
    }
    for (o, &s) in dst.iter_mut().zip(src.iter()) {
        *o += a * s;
    }
}

pub struct SubsamplingWeights {
    pub l0_w: Vec<f32>,
    pub l0_b: Vec<f32>, // (256,1,3,3), (256)
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>, // dw (256,1,3,3)
    pub l3_w: Vec<f32>,
    pub l3_b: Vec<f32>, // pw (256,256,1,1)
    pub l5_w: Vec<f32>,
    pub l5_b: Vec<f32>,
    pub l6_w: Vec<f32>,
    pub l6_b: Vec<f32>,
}

impl SubsamplingWeights {
    pub fn from_tensors(w: &std::collections::HashMap<String, RawTensor>) -> Self {
        Self::from_source(&|n| {
            w.get(n)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing tensor {n}"))
        })
        .expect("weight map is complete")
    }

    /// Tensor-at-a-time variant for a streaming reader.
    pub fn from_source(get: &dyn Fn(&str) -> anyhow::Result<RawTensor>) -> anyhow::Result<Self> {
        let conv = |name: &str| -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let bias_name = format!("{}.bias", name.trim_end_matches(".weight"));
            Ok((get(name)?.to_f32(), get(&bias_name)?.to_f32()))
        };
        let (l0_w, l0_b) = conv("encoder.subsampling.layers.0.weight")?;
        let (l2_w, l2_b) = conv("encoder.subsampling.layers.2.weight")?;
        let (l3_w, l3_b) = conv("encoder.subsampling.layers.3.weight")?;
        let (l5_w, l5_b) = conv("encoder.subsampling.layers.5.weight")?;
        let (l6_w, l6_b) = conv("encoder.subsampling.layers.6.weight")?;
        Ok(Self {
            l0_w,
            l0_b,
            l2_w,
            l2_b,
            l3_w,
            l3_b,
            l5_w,
            l5_b,
            l6_w,
            l6_b,
        })
    }
}

/// Input: (frames, 128) mel rows. Output: (T, 4096) flattened rows where
/// T = subsampled valid length and row t is [c0m0..c0m15, c1m0.., ..., c255m15].
pub fn subsample(sw: &SubsamplingWeights, mel_rows: &[f32], n_valid_frames: usize) -> Vec<f32> {
    let f = mel_rows.len() / cfg::MEL_BINS;
    let m0 = cfg::MEL_BINS;
    // layer 0: (1, F, 128) -> (256, F1, 64)
    let f1 = conv_len(f);
    let m1 = conv_len(m0);
    // First conv (1 -> 256 channels, 3x3 stride 2).  Same tap order and same
    // per-pixel accumulation as the scalar version, but tiled over output pixels
    // (PC) and channels (CO) so the inner loop is a unit-stride AXPY.
    let mut x = {
        let w = &sw.l0_w;
        let b = &sw.l0_b;
        let p = f1 * m1;
        const CO: usize = 16;
        const PC: usize = 256;
        #[cfg(target_arch = "x86_64")]
        let simd = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        let mut out = vec![0f32; cfg::SUB_CHANNELS * p];
        out.par_chunks_mut(CO * p).enumerate().for_each(|(cb, ob)| {
            let co0 = cb * CO;
            let nco = (cfg::SUB_CHANNELS - co0).min(CO);
            for co in 0..nco {
                ob[co * p..(co + 1) * p].fill(b[co0 + co]);
            }
            let mut xs = vec![0f32; 9 * PC];
            let mut pc = 0usize;
            while pc < p {
                let n = (p - pc).min(PC);
                for t in 0..9 {
                    let (ki, kj) = (t / 3, t % 3);
                    let row = &mut xs[t * PC..t * PC + n];
                    row.fill(0.0);
                    for (q, value) in row.iter_mut().enumerate() {
                        let pix = pc + q;
                        let (i, j) = (pix / m1, pix % m1);
                        let ii = 2 * i + ki;
                        let jj = 2 * j + kj;
                        if ii > 0 && ii <= 2 * f && jj > 0 && jj <= 2 * m0 {
                            let (si, sj) = (ii - 1, jj - 1);
                            if si < f && sj < m0 {
                                *value = mel_rows[si * m0 + sj];
                            }
                        }
                    }
                }
                for co in 0..nco {
                    let orow = &mut ob[co * p + pc..co * p + pc + n];
                    for t in 0..9 {
                        axpy(orow, &xs[t * PC..t * PC + n], w[(co0 + co) * 9 + t], simd);
                    }
                }
                pc += PC;
            }
            for co in 0..nco {
                for o in ob[co * p..(co + 1) * p].iter_mut() {
                    *o = o.max(0.0);
                }
            }
        });
        out
    };
    // valid length progression via masks
    let mut vlen = conv_len(n_valid_frames.max(1));
    let cm = cfg::SUB_CHANNELS;

    let dw = |x: &Vec<f32>, w: &Vec<f32>, b: &Vec<f32>, t_in: usize, m_in: usize| -> Vec<f32> {
        let t_out = conv_len(t_in);
        let m_out = conv_len(m_in);
        let mut out = vec![0f32; cm * t_out * m_out];
        out.par_chunks_exact_mut(t_out * m_out)
            .enumerate()
            .for_each(|(c, o)| {
                for i in 0..t_out {
                    for j in 0..m_out {
                        let mut s = b[c];
                        for ki in 0..3isize {
                            let ii = 2 * i as isize + ki - 1;
                            if ii < 0 || ii as usize >= t_in {
                                continue;
                            }
                            for kj in 0..3isize {
                                let jj = 2 * j as isize + kj - 1;
                                if jj < 0 || jj as usize >= m_in {
                                    continue;
                                }
                                s += w[c * 9 + (ki * 3 + kj) as usize]
                                    * x[c * t_in * m_in + ii as usize * m_in + jj as usize];
                            }
                        }
                        o[i * m_out + j] = s;
                    }
                }
            });
        out
    };
    // Pointwise conv = GEMM C[co,p] = W[co,ci] * X[ci,p] with p = t*m.
    // Blocking over co keeps X (the large operand) hot in cache: one pass over X
    // feeds 16 output channels, and the inner loop is a unit-stride f32 AXPY that
    // vectorizes well.  This is ~10x faster than the per-pixel scalar loop.
    let pw = |x: &Vec<f32>, w: &Vec<f32>, b: &Vec<f32>, t: usize, m: usize| -> Vec<f32> {
        let tt = std::time::Instant::now();
        let p = t * m;
        const CO: usize = 16;
        #[cfg(target_arch = "x86_64")]
        let simd = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let simd = false;
        let mut out = vec![0f32; cm * p];
        // Tile the pixel axis as well: a (CO x PC) output tile is 128 KiB, so it
        // stays in L2 across all 256 input channels instead of streaming through
        // L3/DRAM once per input channel.
        const PC: usize = 2048;
        let npc = p.div_ceil(PC);
        out.par_chunks_mut(CO * p).enumerate().for_each(|(cb, ob)| {
            let co0 = cb * CO;
            let nco = (cm - co0).min(CO);
            ob[..nco * p].fill(0.0);
            for pc in 0..npc {
                let p0 = pc * PC;
                let p1 = (p0 + PC).min(p);
                for ci in 0..cm {
                    let xs = ci * p + p0;
                    let xrow = &x[xs..xs + (p1 - p0)];
                    for co in 0..nco {
                        let wv = w[(co0 + co) * cm + ci];
                        if wv != 0.0 {
                            let os = co * p + p0;
                            let orow = &mut ob[os..os + (p1 - p0)];
                            axpy(orow, xrow, wv, simd);
                        }
                    }
                }
            }
            for co in 0..nco {
                let orow = &mut ob[co * p..(co + 1) * p];
                let bv = b[co0 + co];
                for o in orow.iter_mut() {
                    *o = (*o + bv).max(0.0);
                }
            }
        });
        if std::env::var("BARK3_SUBTIME").is_ok() {
            eprintln!(
                "[sub] pw t={t} m={m} p={p}: {:.3}s",
                tt.elapsed().as_secs_f32()
            );
        }
        out
    };

    // mask to vlen after layer0
    mask_time(&mut x, f1, m1, vlen.min(f1));
    let x2 = dw(&x, &sw.l2_w, &sw.l2_b, f1, m1);
    let mut len = conv_len(f1);
    let mut mlen = conv_len(m1);
    vlen = conv_len(vlen);
    let mut x2 = x2;
    mask_time(&mut x2, len, mlen, vlen.min(len));
    let x3 = pw(&x2, &sw.l3_w, &sw.l3_b, len, mlen);
    let x4 = dw(&x3, &sw.l5_w, &sw.l5_b, len, mlen);
    len = conv_len(len);
    mlen = conv_len(mlen);
    vlen = conv_len(vlen);
    let mut x4 = x4;
    mask_time(&mut x4, len, mlen, vlen.min(len));
    let x5 = pw(&x4, &sw.l6_w, &sw.l6_b, len, mlen);
    let t = len;
    // flatten to (T, 4096): row t = [c major, m minor]
    let mut out = vec![0f32; t * cm * mlen];
    for c in 0..cm {
        for i in 0..t {
            for j in 0..mlen {
                out[i * cm * mlen + c * mlen + j] = x5[c * t * mlen + i * mlen + j];
            }
        }
    }
    out
}

fn mask_time(x: &mut [f32], t: usize, m: usize, vlen: usize) {
    // zero time positions >= vlen (torch hidden_states *= channel_mask)
    if vlen >= t {
        return;
    }
    x.par_chunks_exact_mut(t * m).for_each(|ch| {
        for i in vlen..t {
            for j in 0..m {
                ch[i * m + j] = 0.0;
            }
        }
    });
}
