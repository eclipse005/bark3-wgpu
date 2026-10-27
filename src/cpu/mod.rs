//! CPU-resident encoder backend.
//!
//! Mirrors `Encoder::encode_pre` op for op (same kernel order, same f16 weight
//! rounding, same head packing and rel-shift) but runs entirely on the CPU in
//! f32, so it needs no GPU at all. The only intentional deviation is the GEMM
//! accumulation order, which is the machine's own blocking; token output still
//! has to match the reference byte for byte (see `verify.ps1` / `cpuenc`).

pub mod gemm;
pub mod layer;
pub mod ops;

use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::raw::RawTensor;
use crate::weights::WeightSource;
use gemm::{PackW, pack_rows, round_to_f16};
use layer::{CpuLayer, LnW, SEC_COUNT, Scratch};

const HIDDEN: usize = 1024;
const LAYERS: usize = 24;

pub struct CpuEncoder {
    sub_w: PackW,
    sub_b: Vec<f32>,
    layers: Vec<CpuLayer>,
    proj_w: PackW,
    proj_b: Vec<f32>,
}

fn tensor<'a>(w: &'a HashMap<String, RawTensor>, name: &str) -> Result<&'a RawTensor> {
    w.get(name)
        .with_context(|| format!("missing tensor {name}"))
}

/// Checkpoint bf16 -> f16 -> f32: the same rounding the GPU storage path
/// applies, so both backends multiply by identical weight values.
fn f16v(t: &RawTensor) -> Vec<f32> {
    t.to_f32().into_iter().map(round_to_f16).collect()
}

impl CpuEncoder {
    pub fn load(w: &HashMap<String, RawTensor>) -> Result<Self> {
        Self::load_with(&crate::weights::from_map(w))
    }

    /// `src` is asked for one tensor at a time and the result is dropped as soon
    /// as it has been packed, so a streaming reader never has to hold the whole
    /// checkpoint — the difference between a 3.5 GB and a 2.4 GB peak.
    pub fn load_with(src: &dyn WeightSource) -> Result<Self> {
        let get = |name: &str| src.read(name);
        let sub_w = pack_rows(
            &f16v(&get("encoder.subsampling.linear.weight")?),
            1024,
            4096,
        );
        let sub_b = get("encoder.subsampling.linear.bias")?.to_f32();

        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("encoder.layers.{l}.");
            let mut cat = Vec::with_capacity(3 * HIDDEN * HIDDEN);
            for proj in ["q_proj", "k_proj", "v_proj"] {
                cat.extend_from_slice(&f16v(&get(&format!("{p}self_attn.{proj}.weight"))?));
            }
            let qkv = pack_rows(&cat, 3072, HIDDEN);
            drop(cat);
            let relk = f16v(&get(&format!("{p}self_attn.relative_k_proj.weight"))?);
            let o = pack_rows(
                &f16v(&get(&format!("{p}self_attn.o_proj.weight"))?),
                HIDDEN,
                HIDDEN,
            );
            let ff1a = pack_rows(
                &f16v(&get(&format!("{p}feed_forward1.linear1.weight"))?),
                4096,
                HIDDEN,
            );
            let ff1b = pack_rows(
                &f16v(&get(&format!("{p}feed_forward1.linear2.weight"))?),
                HIDDEN,
                4096,
            );
            let ff2a = pack_rows(
                &f16v(&get(&format!("{p}feed_forward2.linear1.weight"))?),
                4096,
                HIDDEN,
            );
            let ff2b = pack_rows(
                &f16v(&get(&format!("{p}feed_forward2.linear2.weight"))?),
                HIDDEN,
                4096,
            );
            let pw1 = pack_rows(
                &f16v(&get(&format!("{p}conv.pointwise_conv1.weight"))?),
                2048,
                HIDDEN,
            );
            let pw2 = pack_rows(
                &f16v(&get(&format!("{p}conv.pointwise_conv2.weight"))?),
                HIDDEN,
                HIDDEN,
            );
            let dw = f16v(&get(&format!("{p}conv.depthwise_conv.weight"))?);

            // BatchNorm folded exactly as `Encoder::load` does it.
            let norm = |suffix: &str| -> Result<Vec<f32>> {
                Ok(get(&format!("{p}conv.norm.{suffix}"))?.to_f32())
            };
            let (gamma, beta, mean, var) = (
                norm("weight")?,
                norm("bias")?,
                norm("running_mean")?,
                norm("running_var")?,
            );
            let mut bn_a = vec![0f32; HIDDEN];
            let mut bn_b = vec![0f32; HIDDEN];
            for c in 0..HIDDEN {
                let a = gamma[c] / (var[c] + 1e-5).sqrt();
                bn_a[c] = a;
                bn_b[c] = beta[c] - mean[c] * a;
            }

            let mut ln = Vec::with_capacity(5);
            for n in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                ln.push(LnW {
                    w: get(&format!("{p}{n}.weight"))?.to_f32(),
                    b: get(&format!("{p}{n}.bias"))?.to_f32(),
                });
            }
            let bu = get(&format!("{p}self_attn.bias_u"))?.to_f32();
            let bv = get(&format!("{p}self_attn.bias_v"))?.to_f32();

            layers.push(CpuLayer::new(
                qkv, relk, o, ff1a, ff1b, ff2a, ff2b, pw1, pw2, dw, bn_a, bn_b, ln, bu, bv,
            ));
        }

        let proj_w = pack_rows(&f16v(&get("encoder_projector.weight")?), 640, HIDDEN);
        let proj_b = get("encoder_projector.bias")?.to_f32();

        Ok(Self {
            sub_w,
            sub_b,
            layers,
            proj_w,
            proj_b,
        })
    }

    /// Run `sub` `(t, 4096)` through `layers` blocks and return either the
    /// pooled `(t, 640)` output or, when `layers < 24`, the residual stream.
    pub fn encode(&self, sub: &[f32], t: usize, layers: usize) -> Vec<f32> {
        let mut s = Scratch::new(t);
        let mut pos_pad = vec![0f32; s.pp * HIDDEN];
        let pos = crate::encoder::pos_embed_f32(t);
        pos_pad[..pos.len()].copy_from_slice(&pos);
        let pos = PackW::from_rows(&pos_pad, s.pp, HIDDEN);

        let mut x = vec![0f32; t * HIDDEN];
        gemm::gemm(sub, &self.sub_w, &mut x, t, 0, HIDDEN);
        ops::add_bias(&mut x, &self.sub_b, t, HIDDEN);

        let upto = layers.min(self.layers.len());
        for (i, block) in self.layers.iter().take(upto).enumerate() {
            block.forward(&mut x, &mut s, &pos);
            // A truncated run keeps the last block's pre-norm residual so it can
            // be compared against the reference `h_b{n}` dumps.
            if i + 1 < upto || upto == self.layers.len() {
                x.copy_from_slice(CpuLayer::normed(&s));
            }
        }

        if upto == self.layers.len() {
            let mut pooler = vec![0f32; t * 640];
            gemm::gemm(&x, &self.proj_w, &mut pooler, t, 0, 640);
            ops::add_bias(&mut pooler, &self.proj_b, t, 640);
            if std::env::var("BARK3_CPU_PROFILE").is_ok() {
                report(&s.sec, t);
            }
            pooler
        } else {
            x
        }
    }
}

fn report(sec: &[f64; SEC_COUNT], t: usize) {
    const NAMES: [&str; SEC_COUNT] = [
        "ln",
        "ff1",
        "qkv",
        "attn_all",
        "out_proj",
        "conv",
        "ff2",
        "pack",
        "  score",
        "  bd",
        "  softmax",
        "  av",
    ];
    let total: f64 = sec.iter().sum();
    eprintln!(
        "[cpu-enc] {total:.3}s over {t} frames ({:.1} ms/layer)",
        total * 1000.0 / 24.0
    );
    for (name, v) in NAMES.iter().zip(sec.iter()) {
        eprintln!(
            "[cpu-enc]   {name:<9} {v:>7.3}s  {:>5.1}%",
            v / total * 100.0
        );
    }
}
