//! Elementwise / reduction kernels for the CPU encoder.
//!
//! Each routine mirrors the WGSL kernel it replaces, including the reduction
//! order (256 strided lanes, tree sum) where that is cheap, so the CPU path
//! stays numerically as close to the GPU path as the GEMM allows.

use rayon::prelude::*;

/// Rows above this count are worth handing to rayon.
const PAR_MIN_ROWS: usize = 8;

/// LayerNorm over `(rows, dim)` with f32 scale/bias.
pub fn layer_norm(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], rows: usize, dim: usize) {
    debug_assert_eq!(dim, 1024, "layer norm reduction is written for 1024");
    let body = |drow: &mut [f32], srow: &[f32]| {
        let x = &srow[..dim];
        let mut rs = [0f32; 256];
        let mut rq = [0f32; 256];
        for (lane, (sv, qv)) in rs.iter_mut().zip(rq.iter_mut()).enumerate() {
            let e0 = x[lane];
            let e1 = x[lane + 256];
            let e2 = x[lane + 512];
            let e3 = x[lane + 768];
            *sv = (e0 + e2) + (e1 + e3);
            *qv = (e0 * e0 + e2 * e2) + (e1 * e1 + e3 * e3);
        }
        let mut s = 128;
        while s > 0 {
            for i in 0..s {
                rs[i] += rs[i + s];
                rq[i] += rq[i + s];
            }
            s >>= 1;
        }
        let inv_n = 1.0 / dim as f32;
        let mean = rs[0] * inv_n;
        let var = rq[0] * inv_n - mean * mean;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for i in 0..dim {
            drow[i] = (x[i] - mean) * inv_std * w[i] + b[i];
        }
    };
    if rows >= PAR_MIN_ROWS {
        dst[..rows * dim]
            .par_chunks_mut(dim)
            .zip(src[..rows * dim].par_chunks(dim))
            .for_each(|(d, s)| body(d, s));
    } else {
        dst[..rows * dim]
            .chunks_mut(dim)
            .zip(src[..rows * dim].chunks(dim))
            .for_each(|(d, s)| body(d, s));
    }
}

/// `x = x / (1 + exp(-x))`.
pub fn silu(x: &mut [f32]) {
    let body = |c: &mut [f32]| {
        for v in c.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    };
    if x.len() >= 4096 {
        x.par_chunks_mut(4096).for_each(body);
    } else {
        body(x);
    }
}

/// `dst[i] = a[i] / (1 + exp(-b[i]))` with the two halves of each row split at
/// `c`.
pub fn glu(src: &[f32], dst: &mut [f32], rows: usize, c: usize) {
    debug_assert!(src.len() >= rows * c * 2);
    let body = |r: usize, drow: &mut [f32]| {
        let row = &src[r * c * 2..r * c * 2 + c * 2];
        for i in 0..c {
            let a = row[i];
            let b = row[c + i];
            drow[i] = a / (1.0 + (-b).exp());
        }
    };
    if rows >= PAR_MIN_ROWS {
        dst[..rows * c]
            .par_chunks_mut(c)
            .enumerate()
            .for_each(|(t, d)| body(t, d));
    } else {
        dst[..rows * c]
            .chunks_mut(c)
            .enumerate()
            .for_each(|(r, d)| body(r, d));
    }
}

/// `a[i] += scale * b[i]`.
pub fn add_scaled(a: &mut [f32], b: &[f32], scale: f32) {
    let body = |ac: &mut [f32], bc: &[f32]| {
        for (x, y) in ac.iter_mut().zip(bc.iter()) {
            *x += scale * *y;
        }
    };
    if a.len() >= 8192 {
        a.par_chunks_mut(8192)
            .zip(b.par_chunks(8192))
            .for_each(|(x, y)| body(x, y));
    } else {
        body(a, b);
    }
}

/// `x[row, c] += bias[c]`.
pub fn add_bias(x: &mut [f32], bias: &[f32], rows: usize, dim: usize) {
    let body = |row: &mut [f32]| {
        for (v, bv) in row.iter_mut().zip(bias.iter().take(dim)) {
            *v += *bv;
        }
    };
    if rows >= PAR_MIN_ROWS {
        x[..rows * dim].par_chunks_mut(dim).for_each(body);
    } else {
        x[..rows * dim].chunks_mut(dim).for_each(body);
    }
}

/// `probs = softmax(scale * (probs + rel_shift(bd)))` for `rows` rows of `t`
/// columns, where row `qi` reads `bd[qi*win + (t-1-(q0+qi)-j0) + k]`.
///
/// Folding the rel-shift add into the max pass removes one read/write round
/// trip over the score matrix, which is the widest buffer in attention.
#[allow(clippy::too_many_arguments)]
pub fn rel_softmax_rows(
    probs: &mut [f32],
    bd: &[f32],
    win: usize,
    rows: usize,
    t: usize,
    q0: usize,
    j0: usize,
    scale: f32,
) {
    let body = |qi: usize, prow: &mut [f32]| {
        let shift = t - 1 - (q0 + qi) - j0;
        let brow = &bd[qi * win..][..win];
        let mut m = f32::NEG_INFINITY;
        for (k, v) in prow.iter_mut().enumerate().take(t) {
            let s = scale * (*v + brow[shift + k]);
            *v = s;
            if s > m {
                m = s;
            }
        }
        let mut part = [0f32; 256];
        for (lane, slot) in part.iter_mut().enumerate() {
            let mut acc = 0f32;
            let mut j = lane;
            while j < t {
                let e = (prow[j] - m).exp();
                prow[j] = e;
                acc += e;
                j += 256;
            }
            *slot = acc;
        }
        let mut s = 128;
        while s > 0 {
            for i in 0..s {
                part[i] += part[i + s];
            }
            s >>= 1;
        }
        let sum = part[0];
        for v in prow[..t].iter_mut() {
            *v /= sum;
        }
    };
    if rows >= PAR_MIN_ROWS {
        probs[..rows * t]
            .par_chunks_mut(t)
            .enumerate()
            .for_each(|(qi, r)| body(qi, r));
    } else {
        probs[..rows * t]
            .chunks_mut(t)
            .enumerate()
            .for_each(|(qi, r)| body(qi, r));
    }
}

/// Depthwise conv (k=9, zero pad 4) followed by inference BatchNorm and SiLU.
///
/// `x` and `dst` are `(rows, dim)`; `dw` is `(dim, 9)`. Only rows `< rows` are
/// produced — the GPU's padded rows never influence real frames.
pub fn dwconv_bn_silu(
    src: &[f32],
    dw: &[f32],
    bn_a: &[f32],
    bn_b: &[f32],
    dst: &mut [f32],
    rows: usize,
    dim: usize,
) {
    let body = |t: usize, drow: &mut [f32]| {
        for (c, out) in drow.iter_mut().enumerate().take(dim) {
            let mut acc = 0f32;
            for i in 0..9usize {
                let shifted = t + i;
                if shifted < 4 {
                    continue;
                }
                let tt = shifted - 4;
                if tt >= rows {
                    continue;
                }
                acc += src[tt * dim + c] * dw[c * 9 + i];
            }
            let y = acc * bn_a[c] + bn_b[c];
            *out = y / (1.0 + (-y).exp());
        }
    };
    if rows >= PAR_MIN_ROWS {
        dst[..rows * dim]
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(t, d)| body(t, d));
    } else {
        dst[..rows * dim]
            .chunks_mut(dim)
            .enumerate()
            .for_each(|(t, d)| body(t, d));
    }
}

/// Split `qkv` (rows, 3072) into head-major `q + bias_u`, `q + bias_v` (f32)
/// and `k`, `v` rounded through f16, matching the GPU's head packing.
#[allow(clippy::too_many_arguments)]
pub fn pack_attention_inputs(
    qkv: &[f32],
    rows: usize,
    bu: &[f32],
    bv: &[f32],
    qu: &mut [f32],
    qv: &mut [f32],
    kh: &mut [f32],
    vh: &mut [f32],
) {
    const HEADS: usize = 8;
    const HDIM: usize = 128;
    let span = HEADS * rows * HDIM;
    let body = |i: usize, q_u: &mut [f32], q_v: &mut [f32], k_h: &mut [f32], v_h: &mut [f32]| {
        let h = i / rows;
        let q = i % rows;
        for j in 0..HDIM {
            let b = h * HDIM + j;
            q_u[j] = qkv[q * 3072 + b] + bu[b];
            q_v[j] = qkv[q * 3072 + b] + bv[b];
            k_h[j] = super::gemm::round_to_f16(qkv[q * 3072 + 1024 + b]);
            v_h[j] = super::gemm::round_to_f16(qkv[q * 3072 + 2048 + b]);
        }
    };
    if rows >= PAR_MIN_ROWS {
        qu[..span]
            .par_chunks_mut(HDIM)
            .zip(qv[..span].par_chunks_mut(HDIM))
            .zip(kh[..span].par_chunks_mut(HDIM))
            .zip(vh[..span].par_chunks_mut(HDIM))
            .enumerate()
            .for_each(|(i, (((a, b), c), d))| body(i, a, b, c, d));
    } else {
        qu[..span]
            .chunks_mut(HDIM)
            .zip(qv[..span].chunks_mut(HDIM))
            .zip(kh[..span].chunks_mut(HDIM))
            .zip(vh[..span].chunks_mut(HDIM))
            .enumerate()
            .for_each(|(i, (((a, b), c), d))| body(i, a, b, c, d));
    }
}
