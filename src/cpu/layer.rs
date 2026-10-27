//! One conformer block of the CPU encoder, mirroring `Encoder::encode_pre`.

use rayon::prelude::*;

use super::gemm::{PackW, gemm, gemm_split_k};
use super::ops;

const HIDDEN: usize = 1024;
const HEADS: usize = 8;
const HDIM: usize = 128;
/// Query rows per attention block. Small enough that the score matrix stays in
/// L2, large enough to fill every core in the per-head GEMMs.
const BQ: usize = 256;
/// Panel granularity of `PackW`, so a column window can start anywhere.
const PANEL: usize = 16;

pub struct LnW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct CpuLayer {
    qkv: PackW,
    relk: Vec<f32>,
    o: PackW,
    ff1a: PackW,
    ff1b: PackW,
    ff2a: PackW,
    ff2b: PackW,
    pw1: PackW,
    pw2: PackW,
    dw: Vec<f32>,
    bn_a: Vec<f32>,
    bn_b: Vec<f32>,
    ln: Vec<LnW>,
    bu: Vec<f32>,
    bv: Vec<f32>,
}

/// Section indices for the optional profile.
pub const SEC_LN: usize = 0;
pub const SEC_FF1: usize = 1;
pub const SEC_QKV: usize = 2;
pub const SEC_ATTN: usize = 3;
pub const SEC_OUT: usize = 4;
pub const SEC_CONV: usize = 5;
pub const SEC_FF2: usize = 6;
pub const SEC_PACK: usize = 7;
pub const SEC_ASCORE: usize = 8;
pub const SEC_ABD: usize = 9;
pub const SEC_ASOFT: usize = 10;
pub const SEC_AAV: usize = 11;
pub const SEC_COUNT: usize = 12;

pub struct Scratch {
    pub t: usize,
    pub pp: usize,
    normed: Vec<f32>,
    ffb: Vec<f32>,
    tmp: Vec<f32>,
    tmp2: Vec<f32>,
    qkv: Vec<f32>,
    qu: Vec<f32>,
    qv: Vec<f32>,
    kh: Vec<f32>,
    vh: Vec<f32>,
    relk_t: Vec<f32>,
    attnout: Vec<f32>,
    cb1: Vec<f32>,
    cb2: Vec<f32>,
    probs: Vec<f32>,
    bd: Vec<f32>,
    attn_h: Vec<f32>,
    /// partial sums for the split-k AV GEMM
    av_parts: Vec<f32>,
    kh_pack: Vec<PackW>,
    vh_pack: Vec<PackW>,
    rkh_pack: Vec<PackW>,
    pub sec: [f64; SEC_COUNT],
}

impl Scratch {
    pub fn new(t: usize) -> Self {
        let pp = crate::encoder::pad128(2 * t - 1);
        let packs = || (0..HEADS).map(|_| PackW::empty()).collect::<Vec<_>>();
        Self {
            t,
            pp,
            normed: vec![0.0; t * HIDDEN],
            ffb: vec![0.0; t * 4096],
            tmp: vec![0.0; t * HIDDEN],
            tmp2: vec![0.0; t * HIDDEN],
            qkv: vec![0.0; t * 3072],
            qu: vec![0.0; HEADS * t * HDIM],
            qv: vec![0.0; HEADS * t * HDIM],
            kh: vec![0.0; HEADS * t * HDIM],
            vh: vec![0.0; HEADS * t * HDIM],
            relk_t: vec![0.0; HIDDEN * pp],
            attnout: vec![0.0; t * HIDDEN],
            cb1: vec![0.0; t * 2048],
            cb2: vec![0.0; t * HIDDEN],
            probs: vec![0.0; BQ * t],
            // Window width is at most `t + BQ` rounded out to both panel edges.
            bd: vec![0.0; BQ * (t + BQ + 2 * PANEL)],
            attn_h: vec![0.0; BQ * HDIM],
            av_parts: vec![0.0; rayon::current_num_threads().max(1) * BQ * HDIM],
            kh_pack: packs(),
            vh_pack: packs(),
            rkh_pack: packs(),
            sec: [0.0; SEC_COUNT],
        }
    }
}

impl CpuLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv: PackW,
        relk: Vec<f32>,
        o: PackW,
        ff1a: PackW,
        ff1b: PackW,
        ff2a: PackW,
        ff2b: PackW,
        pw1: PackW,
        pw2: PackW,
        dw: Vec<f32>,
        bn_a: Vec<f32>,
        bn_b: Vec<f32>,
        ln: Vec<LnW>,
        bu: Vec<f32>,
        bv: Vec<f32>,
    ) -> Self {
        Self {
            qkv,
            relk,
            o,
            ff1a,
            ff1b,
            ff2a,
            ff2b,
            pw1,
            pw2,
            dw,
            bn_a,
            bn_b,
            ln,
            bu,
            bv,
        }
    }

    /// `x` is `(t, 1024)` and is updated in place with the full block
    /// (FF1 -> attention -> conv -> FF2 -> final norm), exactly the order the
    /// GPU encoder uses.
    pub fn forward(&self, x: &mut [f32], s: &mut Scratch, pos: &PackW) {
        let t = s.t;
        let pp = s.pp;
        let scale = (HDIM as f32).powf(-0.5);
        let Scratch {
            normed,
            ffb,
            tmp,
            tmp2,
            qkv,
            qu,
            qv,
            kh,
            vh,
            relk_t,
            attnout,
            cb1,
            cb2,
            probs,
            bd,
            attn_h,
            av_parts,
            kh_pack,
            vh_pack,
            rkh_pack,
            sec,
            ..
        } = s;

        // ── FF1: h += 0.5 * ff2(silu(ff1(ln1(h))))
        let t0 = std::time::Instant::now();
        ops::layer_norm(x, normed, &self.ln[0].w, &self.ln[0].b, t, HIDDEN);
        sec[SEC_LN] += t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        gemm(normed, &self.ff1a, ffb, t, 0, 4096);
        ops::silu(&mut ffb[..t * 4096]);
        gemm(ffb, &self.ff1b, tmp, t, 0, HIDDEN);
        ops::add_scaled(x, tmp, 0.5);
        sec[SEC_FF1] += t0.elapsed().as_secs_f64();

        // ── attention
        let t0 = std::time::Instant::now();
        ops::layer_norm(x, normed, &self.ln[1].w, &self.ln[1].b, t, HIDDEN);
        sec[SEC_LN] += t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        gemm(normed, &self.qkv, qkv, t, 0, 3072);
        sec[SEC_QKV] += t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        ops::pack_attention_inputs(qkv, t, &self.bu, &self.bv, qu, qv, kh, vh);
        // `k` is consumed as the weight of the score GEMM, so it packs as an
        // (n = t, k = 128) row-major matrix; `v` is consumed on the k side of
        // the AV GEMM, so it packs as (k = t, n = 128).
        kh_pack
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, w)| w.refill_rows(&kh[h * t * HDIM..], t, HDIM, false));
        vh_pack
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, w)| w.refill_cols(&vh[h * t * HDIM..], HDIM, 0, HDIM, t, false));
        // Relative-position keys: relk_t[d][p] = relk[d] . pos[p], then rounded
        // through f16 like the GPU's f2h_mix does.
        gemm(&self.relk, pos, relk_t, HIDDEN, 0, pp);
        rkh_pack
            .par_iter_mut()
            .enumerate()
            .for_each(|(h, w)| w.refill_cols(&relk_t[h * HDIM * pp..], pp, 0, pp, HDIM, true));
        sec[SEC_PACK] += t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let mut q0 = 0usize;
        while q0 < t {
            let rows = (t - q0).min(BQ);
            let jmin = t - 1 - (q0 + rows - 1);
            let jmax = (2 * t - 2 - q0).min(pp - 1);
            let j0 = (jmin / PANEL) * PANEL;
            let j1 = ((jmax + PANEL) / PANEL * PANEL).min(pp);
            let win = j1 - j0;
            debug_assert!(win <= t + BQ + 2 * PANEL);
            for h in 0..HEADS {
                let t1 = std::time::Instant::now();
                let a = &qu[h * t * HDIM + q0 * HDIM..];
                gemm(a, &kh_pack[h], &mut probs[..rows * t], rows, 0, t);
                sec[SEC_ASCORE] += t1.elapsed().as_secs_f64();

                let t1 = std::time::Instant::now();
                let a = &qv[h * t * HDIM + q0 * HDIM..];
                gemm(a, &rkh_pack[h], &mut bd[..rows * win], rows, j0, win);
                sec[SEC_ABD] += t1.elapsed().as_secs_f64();

                let t1 = std::time::Instant::now();
                // rel_shift + scale + softmax over the real frames, one pass.
                let bd_ref: &[f32] = bd;
                ops::rel_softmax_rows(&mut probs[..rows * t], bd_ref, win, rows, t, q0, j0, scale);
                sec[SEC_ASOFT] += t1.elapsed().as_secs_f64();

                let t1 = std::time::Instant::now();
                gemm_split_k(
                    &probs[..rows * t],
                    &vh_pack[h],
                    &mut attn_h[..rows * HDIM],
                    av_parts,
                    rows,
                    0,
                    HDIM,
                );
                sec[SEC_AAV] += t1.elapsed().as_secs_f64();
                for qi in 0..rows {
                    let dst = &mut attnout[(q0 + qi) * HIDDEN + h * HDIM..][..HDIM];
                    dst.copy_from_slice(&attn_h[qi * HDIM..][..HDIM]);
                }
            }
            q0 += rows;
        }
        sec[SEC_ATTN] += t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        gemm(attnout, &self.o, tmp, t, 0, HIDDEN);
        ops::add_scaled(x, tmp, 1.0);
        sec[SEC_OUT] += t0.elapsed().as_secs_f64();

        // ── conv module
        let t0 = std::time::Instant::now();
        ops::layer_norm(x, normed, &self.ln[2].w, &self.ln[2].b, t, HIDDEN);
        gemm(normed, &self.pw1, cb1, t, 0, 2048);
        ops::glu(cb1, cb2, t, HIDDEN);
        ops::dwconv_bn_silu(cb2, &self.dw, &self.bn_a, &self.bn_b, tmp, t, HIDDEN);
        gemm(tmp, &self.pw2, tmp2, t, 0, HIDDEN);
        ops::add_scaled(x, tmp2, 1.0);
        sec[SEC_CONV] += t0.elapsed().as_secs_f64();

        // ── FF2: h += 0.5 * ff2(silu(ff1(ln3(h))))
        let t0 = std::time::Instant::now();
        ops::layer_norm(x, normed, &self.ln[3].w, &self.ln[3].b, t, HIDDEN);
        gemm(normed, &self.ff2a, ffb, t, 0, 4096);
        ops::silu(&mut ffb[..t * 4096]);
        gemm(ffb, &self.ff2b, tmp, t, 0, HIDDEN);
        ops::add_scaled(x, tmp, 0.5);
        sec[SEC_FF2] += t0.elapsed().as_secs_f64();

        // ── block-final norm_out. It is written to the scratch instead of `x`
        // so a caller can still see the pre-norm residual (the reference dump
        // is taken before the swap).
        let t0 = std::time::Instant::now();
        ops::layer_norm(x, normed, &self.ln[4].w, &self.ln[4].b, t, HIDDEN);
        sec[SEC_LN] += t0.elapsed().as_secs_f64();
    }

    /// The `norm_out` result of the most recent [`CpuLayer::forward`].
    pub fn normed(s: &Scratch) -> &[f32] {
        let t = s.t;
        &s.normed[..t * HIDDEN]
    }
}
