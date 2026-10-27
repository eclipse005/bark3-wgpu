//! GPU encoder: conv-subsampled input -> 24 conformer layers -> pooler output.
//!
//! Dispatch plan per layer (buffers preallocated per call, T padded to 128):
//!   LN -> qkv GEMM -> q_u/q_v head-pack(+bias u/v) -> relk GEMM -> head-pack
//!   k,v -> content scores GEMM -> bd GEMM -> rel-add(scale) -> softmax
//!   -> AV GEMM -> flat pack -> o GEMM -> add
//!   LN -> ff1 GEMM -> silu -> ff2 GEMM (weights pre-halved) -> add
//!   LN -> pw1 GEMM -> glu -> dwconv9 -> bn+silu -> pw2 GEMM -> add
//! Final: LN -> enc_proj GEMM (+bias) -> pooler readback.

// Kernel wrappers intentionally mirror shader bindings and buffer strides;
// replacing them with ad-hoc parameter structs would obscure launch sites.
#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use bytemuck::Pod;
use bytemuck::Zeroable;

use crate::bark::LowBitTensor;
use crate::cfg;
use crate::gpu::{BulkUpload, Gpu};
use crate::raw::RawTensor;
use crate::shaders;
use crate::weights::WeightSource;

mod gemm;
mod prof;
mod shader_sources;

pub use gemm::{gemm_mix, gemm_mix_4x8, gemm_mix_8x4, gemm_mix_bm64, gemm_mix_bm64_lb};
use prof::Prof;
use shader_sources::*;

pub fn pad128(n: usize) -> usize {
    (n + 127) & !127
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GDims {
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    bsa: u32,
    bsb: u32,
    bsc: u32,
    beta: u32,
    row0: u32,
    lda: u32,
    _pad: u32,
    /// First word of the code array (`0` when the weight is f16).
    lev_words: u32,
    /// Levels per row (`0` when the weight is f16).
    k_lev: u32,
    /// Words per row in the code array (`0` when the weight is f16).
    codes_ld: u32,
    lowbit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U7 {
    a: u32,
    b: u32,
    c: u32,
    d: f32,
    e: u32,
    f: u32,
    g: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U5f {
    a: u32,
    b: u32,
    c: f32,
    d: u32,
}

pub struct EncoderWeights {
    pub sub_w: Weight,       // (1024, 4096), f16 or low-bit
    pub sub_b: wgpu::Buffer, // (1024) f16
    pub layers: Vec<LayerWeights>,
    pub proj_w: wgpu::Buffer, // (640, 1024)
    pub proj_b: wgpu::Buffer, // (640)
}

/// Where a low-bit weight's metadata lives, all relative to the same buffer:
/// `[levels: n rows x 4 f16][codes: n rows x codes_ld u32]`.
#[derive(Debug, Clone, Copy)]
pub struct LowBit {
    pub lev_words: u32,
    pub k_lev: u32,
    pub codes_ld: u32,
}

/// A dense GEMM weight: either f16 halves or 4-bit codes with a level table.
#[derive(Clone)]
pub struct Weight {
    pub buf: wgpu::Buffer,
    pub lb: Option<LowBit>,
}

impl Weight {
    /// Weights that stay in the f16 layout (the bf16 release, the decoder, activations).
    pub fn f16(buf: wgpu::Buffer) -> Self {
        Self { buf, lb: None }
    }

    pub fn lowbit(&self) -> Option<LowBit> {
        self.lb
    }
}

pub struct LayerWeights {
    pub qkv: Weight,           // (3072, 1024) fused q|k|v
    pub relk: Weight,          // (1024, 1024)
    pub o: Weight,             // (1024, 1024)
    pub ff1a: Weight,          // ffn1.linear1 (4096, 1024)
    pub ff1b: Weight,          // ffn1.linear2 (1024, 4096) pre-scaled 0.5
    pub ff2a: Weight,          // ffn2.linear1 (4096, 1024)
    pub ff2b: Weight,          // ffn2.linear2 (1024, 4096) pre-scaled 0.5
    pub pw1: Weight,           // (2048, 1024)
    pub pw2: Weight,           // (1024, 1024)
    pub dw: wgpu::Buffer,      // (1024, 9) f16
    pub bn_ab: wgpu::Buffer,   // f32 [a; b]
    pub ln: [wgpu::Buffer; 5], // f16 [w; b] each (2*512 words)
    pub bu: wgpu::Buffer,      // bias_u (8,128) f16
    pub bv: wgpu::Buffer,
}

pub struct Encoder {
    pub w: EncoderWeights,
    gemm: wgpu::ComputePipeline,
    gemm_av: wgpu::ComputePipeline,
    gemm_bias: wgpu::ComputePipeline,
    ln: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    pub add_scaled: wgpu::ComputePipeline,
    pub gemm_f32: wgpu::ComputePipeline,
    pub gemm_f32_tb: wgpu::ComputePipeline,
    pub gemm_f32_bm64: wgpu::ComputePipeline,
    /// 4-bit codes + per-row levels; see [`LowBit`].
    pub gemm_lb_bm64: wgpu::ComputePipeline,
    pub gemm_f32_tb_bm64: wgpu::ComputePipeline,
    pub gemm_f32_4x8: wgpu::ComputePipeline,
    pub gemm_f32_tb_4x8: wgpu::ComputePipeline,
    pub gemm_f32_8x4: wgpu::ComputePipeline,
    pub gemm_f32_tb_8x4: wgpu::ComputePipeline,
    pub ln_f32: wgpu::ComputePipeline,
    pub silu_f32: wgpu::ComputePipeline,
    pub glu_f32: wgpu::ComputePipeline,
    pub add_scaled_f32: wgpu::ComputePipeline,
    pub f2h_mix: wgpu::ComputePipeline,
    pub add_head_f32b: wgpu::ComputePipeline,
    pub add_bias_f32: wgpu::ComputePipeline,
    pub h2f_f32b: wgpu::ComputePipeline,
    pub dwconv_f32: wgpu::ComputePipeline,
    pub bn_silu_f32: wgpu::ComputePipeline,
    pub add_head_f32: wgpu::ComputePipeline,
    pub rel_add_f32: wgpu::ComputePipeline,
    pub softmax_f32: wgpu::ComputePipeline,
    /// Fused bias+softmax; see `Ctx::add_softmax_f32`.
    pub add_softmax_f32: wgpu::ComputePipeline,
    pub h2f_f32: wgpu::ComputePipeline,
    add_bias: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    add_head: wgpu::ComputePipeline,
    f2h: wgpu::ComputePipeline,
    h2f: wgpu::ComputePipeline,
    rel_add: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    dwconv: wgpu::ComputePipeline,
    bn_silu: wgpu::ComputePipeline,
    gemm_bgl: wgpu::BindGroupLayout,
    gemm_av_bgl: wgpu::BindGroupLayout,
    gemm_bias_bgl: wgpu::BindGroupLayout,
    ln_bgl: wgpu::BindGroupLayout,
    u4_bgl: wgpu::BindGroupLayout,
    s2_bgl: wgpu::BindGroupLayout,
    s2io_bgl: wgpu::BindGroupLayout,
    s1_bgl: wgpu::BindGroupLayout,
    ah_bgl: wgpu::BindGroupLayout,
    uni: wgpu::Buffer, // 128 slots * 256B
    /// Present only when `BARK3_ENC_PROFILE` is set; see `encoder::prof`.
    pub prof: Option<Prof>,
    pub gpu: Gpu,
}

/// Longest attention row the fused bias+softmax kernel can hold in workgroup
/// memory (16 KiB of f32). Bigger caches drop occupancy and lose the win;
/// longer utterances fall back to the two-kernel sequence.
/// Must match the `cache` array in `P_ADD_SOFTMAX_F32`.
pub const MAX_ADD_SOFTMAX_ROW: usize = 4096;

const UNI_SLOT: u64 = 256;
const UNI_SLOTS: u64 = 240;

pub struct EncOpts {
    pub dump_layer: Option<usize>, // golden check: readback this block's output
}

/// Names the bm64 GEMM by its output shape so the profiler separates the
/// per-layer projections instead of lumping them together.
fn gemm_label(n: u32, k: u32) -> &'static str {
    match (n, k) {
        (3072, 1024) => "gemm_qkv",
        (4096, 1024) => "gemm_ff1",
        (1024, 4096) => "gemm_ff2",
        (1024, 1024) => "gemm_attn_o",
        (2048, 1024) => "gemm_pw1",
        (1024, 2048) => "gemm_pw2",
        (640, 1024) => "gemm_proj",
        _ => "gemm_f32",
    }
}

pub fn pos_embed_f32(t: usize) -> Vec<f32> {
    let p = 2 * t - 1;
    let mut out = vec![0f32; p * cfg::HIDDEN];
    let mut inv_freq = vec![0f32; cfg::HIDDEN / 2];
    for (i, f) in inv_freq.iter_mut().enumerate() {
        *f = 1.0 / 10000f32.powf((2.0 * i as f32) / cfg::HIDDEN as f32);
    }
    for (pi, row) in out.chunks_exact_mut(cfg::HIDDEN).enumerate() {
        let pos = (p as isize - 1) / 2 - pi as isize; // arange(T-1, -T, -1)
        let pos = pos as f32;
        for i in 0..cfg::HIDDEN / 2 {
            let f = pos * inv_freq[i];
            row[2 * i] = f.sin();
            row[2 * i + 1] = f.cos();
        }
    }
    out
}

// ───────────────────────────── bind group helpers ─────────────────────────────

fn bgl(device: &wgpu::Device, label: &str, entries: &[(u32, EntryKind)]) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries
            .iter()
            .map(|(b, k)| wgpu::BindGroupLayoutEntry {
                binding: *b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match k {
                    EntryKind::Ro => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    EntryKind::Rw => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    EntryKind::UniDyn => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(64),
                    },
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    })
}

enum EntryKind {
    Ro,
    Rw,
    UniDyn,
}

use EntryKind::{Ro, Rw, UniDyn};

/// Upload a 4-bit weight: `[levels: rows x 4 f16][codes: rows x codes_ld u32]`.
///
/// The layout is what `gemm_mix_bm64_lb` expects: the level table first (two
/// words per output row) so the kernel can find it from one base offset, then
/// the codes, eight 4-bit weights per word along the input dimension.
fn upload_lowbit(
    gpu: &Gpu,
    up: &mut BulkUpload<'_>,
    label: &str,
    lb: &LowBitTensor,
) -> Result<Weight> {
    let rows = lb.rows;
    let codes_ld = lb.cols.div_ceil(8) as u32;
    anyhow::ensure!(
        lb.codes.len() == rows * codes_ld as usize && lb.levels.len() == rows * 4,
        "{label}: low-bit layout mismatch"
    );
    let words = rows as u64 * 2 + rows as u64 * codes_ld as u64;
    let buf = gpu.storage(label, words * 4);
    up.upload(&buf, bytemuck::cast_slice(&lb.levels))?;
    up.upload_at(&buf, rows as u64 * 2 * 4, bytemuck::cast_slice(&lb.codes))?;
    Ok(Weight {
        buf,
        lb: Some(LowBit {
            lev_words: 0,
            k_lev: lb.k,
            codes_ld,
        }),
    })
}

/// One dense GEMM weight, in whichever layout the checkpoint ships.
#[allow(clippy::too_many_arguments)]
fn dense(
    gpu: &Gpu,
    up: &mut BulkUpload<'_>,
    src: &dyn WeightSource,
    prefix: &str,
    layer: usize,
    label: &str,
    name: &str,
    n: u64,
    k: u64,
) -> Result<Weight> {
    let full = format!("{prefix}{name}");
    if let Some(lb) = src.read_lowbit(&full)? {
        return upload_lowbit(gpu, up, &format!("l{layer}{label}"), &lb);
    }
    let b = gpu.storage(&format!("l{layer}{label}"), n * k * 2);
    up.upload(&b, bytemuck::cast_slice(&src.read(&full)?.to_f16_words()))?;
    Ok(Weight::f16(b))
}

/// The fused q|k|v matrix, in either layout (one buffer, three row blocks).
fn load_qkv(
    gpu: &Gpu,
    up: &mut BulkUpload<'_>,
    src: &dyn WeightSource,
    l: usize,
    p: &str,
) -> Result<Weight> {
    let names: Vec<String> = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        .iter()
        .map(|x| format!("{p}{x}.weight"))
        .collect();
    let low: Vec<_> = names
        .iter()
        .map(|n| src.read_lowbit(n))
        .collect::<Result<Vec<_>>>()?;
    if low.iter().all(Option::is_some) {
        let parts: Vec<std::sync::Arc<LowBitTensor>> =
            low.into_iter().map(Option::unwrap).collect();
        let k = parts[0].k;
        anyhow::ensure!(
            parts.iter().all(|t| t.k == k),
            "fused q/k/v must share one level count"
        );
        let codes_ld = parts[0].cols.div_ceil(8) as u32;
        let rows = parts[0].rows * parts.len();
        let buf = gpu.storage(
            &format!("l{l}qkv"),
            (rows as u64 * 2 + rows as u64 * codes_ld as u64) * 4,
        );
        for (i, part) in parts.iter().enumerate() {
            let base = i as u64 * part.rows as u64;
            up.upload_at(&buf, base * 2 * 4, bytemuck::cast_slice(&part.levels))?;
            up.upload_at(
                &buf,
                (rows as u64 * 2 + base * codes_ld as u64) * 4,
                bytemuck::cast_slice(&part.codes),
            )?;
        }
        return Ok(Weight {
            buf,
            lb: Some(LowBit {
                lev_words: 0,
                k_lev: k,
                codes_ld,
            }),
        });
    }
    anyhow::ensure!(
        low.iter().all(Option::is_none),
        "fused q/k/v mixes low-bit and f16 storage"
    );
    let buf = gpu.storage(&format!("l{l}qkv"), 3 * 1024 * 1024 * 2);
    for (i, name) in names.iter().enumerate() {
        up.upload_at(
            &buf,
            (i * 1024 * 1024 * 2) as u64,
            bytemuck::cast_slice(&src.read(name)?.to_f16_words()),
        )?;
    }
    Ok(Weight::f16(buf))
}

impl Encoder {
    pub fn load(gpu: Gpu, w: &std::collections::HashMap<String, RawTensor>) -> Result<Self> {
        Self::load_with(gpu, &crate::weights::from_map(w))
    }

    /// Loads from whichever official layout the checkpoint uses: bf16
    /// `model.safetensors`, or the `model.bark` container whose encoder linears
    /// stay 4-bit. `src.prefetch()` decodes the container in parallel first.
    pub fn load_with(gpu: Gpu, src: &dyn WeightSource) -> Result<Self> {
        let _stage = crate::browser::stage("enc");
        crate::browser::log("[enc] prefetch");
        src.prefetch()?;
        crate::browser::log("[enc] prefetched");
        // A read failure here means the checkpoint is truncated; the previous
        // map-based loader panicked on a missing tensor for the same reason.
        let t = |n: &str| src.read(n).unwrap_or_else(|e| panic!("tensor {n}: {e}"));
        let mut up = gpu.uploader();
        let words16 = |n: &str| -> Vec<u32> { t(n).to_f16_words() };

        let sub_w_name = "encoder.subsampling.linear.weight";
        let sub_w = match src.read_lowbit(sub_w_name)? {
            Some(lb) => upload_lowbit(&gpu, &mut up, "sub_w", &lb)?,
            None => {
                let b = gpu.storage("sub_w", 1024 * 4096 * 2);
                up.upload(&b, bytemuck::cast_slice(&words16(sub_w_name)))?;
                Weight::f16(b)
            }
        };
        let sub_b = gpu.storage("sub_b", 1024 * 4);
        up.upload(
            &sub_b,
            bytemuck::cast_slice(&t("encoder.subsampling.linear.bias").to_f32()),
        )?;

        let mut layers = Vec::with_capacity(cfg::LAYERS);
        for l in 0..cfg::LAYERS {
            let p = format!("encoder.layers.{l}.");
            let qkv = load_qkv(&gpu, &mut up, src, l, &p)?;

            // Dense GEMM weight: f16 halves, or 4-bit codes + levels when the
            // checkpoint ships them (see LowBit).
            let relk = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "relk",
                "self_attn.relative_k_proj.weight",
                1024,
                1024,
            )?;
            let o = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "o",
                "self_attn.o_proj.weight",
                1024,
                1024,
            )?;
            let ff1a = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "ff1a",
                "feed_forward1.linear1.weight",
                4096,
                1024,
            )?;
            let ff2a = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "ff2a",
                "feed_forward2.linear1.weight",
                4096,
                1024,
            )?;
            let pw1 = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "pw1",
                "conv.pointwise_conv1.weight",
                2048,
                1024,
            )?;
            let pw2 = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "pw2",
                "conv.pointwise_conv2.weight",
                1024,
                1024,
            )?;
            let dw = gpu.storage(&format!("l{l}dw"), 1024 * 9 * 2);
            up.upload(
                &dw,
                bytemuck::cast_slice(&words16(&format!("{p}conv.depthwise_conv.weight"))),
            )?;
            let mut mk_f32uv = |label: &str, name: &str| -> anyhow::Result<wgpu::Buffer> {
                let b = gpu.storage(&format!("l{l}{label}"), 8 * 128 * 4);
                up.upload(&b, bytemuck::cast_slice(&t(&format!("{p}{name}")).to_f32()))?;
                Ok(b)
            };
            let bu = mk_f32uv("bu", "self_attn.bias_u")?;
            let bv = mk_f32uv("bv", "self_attn.bias_v")?;
            // second linears pre-scaled by 0.5 (residual halving folded into weights)
            let ff1b = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "ff1b",
                "feed_forward1.linear2.weight",
                1024,
                4096,
            )?;
            let ff2b = dense(
                &gpu,
                &mut up,
                src,
                &p,
                l,
                "ff2b",
                "feed_forward2.linear2.weight",
                1024,
                4096,
            )?;
            // BatchNorm folded: a = w/sqrt(var+eps), b = bias - mean*a
            let g = |suffix: &str| t(&format!("{p}conv.norm.{suffix}")).to_f32();
            let (wgt, bias, mean, var) =
                (g("weight"), g("bias"), g("running_mean"), g("running_var"));
            let mut ab = vec![0f32; 2048];
            for c in 0..1024 {
                let a = wgt[c] / (var[c] + 1e-5).sqrt();
                ab[c] = a;
                ab[1024 + c] = bias[c] - mean[c] * a;
            }
            let bn_ab = gpu.storage(&format!("l{l}bn"), 2048 * 4);
            up.upload(&bn_ab, bytemuck::cast_slice(&ab))?;
            let mut ln = Vec::new();
            for n in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                let mut wb = t(&format!("{p}{n}.weight")).to_f32();
                wb.extend_from_slice(&t(&format!("{p}{n}.bias")).to_f32());
                let b = gpu.storage(&format!("l{l}{n}"), wb.len() as u64 * 4);
                up.upload(&b, bytemuck::cast_slice(&wb))?;
                ln.push(b);
            }
            layers.push(LayerWeights {
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
                bn_ab,
                ln: [
                    ln[0].clone(),
                    ln[1].clone(),
                    ln[2].clone(),
                    ln[3].clone(),
                    ln[4].clone(),
                ],
                bu,
                bv,
            })
        }
        let proj_w = gpu.storage("proj_w", 640 * 1024 * 2);
        up.upload(
            &proj_w,
            bytemuck::cast_slice(&words16("encoder_projector.weight")),
        )?;
        let proj_b = gpu.storage("proj_b", 640 * 4);
        up.upload(
            &proj_b,
            bytemuck::cast_slice(&t("encoder_projector.bias").to_f32()),
        )?;
        up.finish()?;

        // bind group layouts (uniform entries dynamic -> slot buffer)
        let gemm_bgl = bgl(
            &gpu.device,
            "gemm",
            &[(0, Ro), (1, Ro), (2, Rw), (3, UniDyn)],
        );
        let gemm_bias_bgl = bgl(
            &gpu.device,
            "gemm_bias",
            &[(0, Ro), (1, Ro), (2, Rw), (3, UniDyn), (4, Ro)],
        );
        let ln_bgl = bgl(
            &gpu.device,
            "ln",
            &[(0, Ro), (1, Ro), (2, Ro), (3, UniDyn), (4, Rw)],
        );
        let u4_bgl = bgl(&gpu.device, "u4", &[(0, Ro), (1, Ro), (2, Rw), (3, UniDyn)]);
        let s2_bgl = bgl(&gpu.device, "s2", &[(0, Ro), (1, Rw), (2, UniDyn)]);
        let s2io_bgl = bgl(&gpu.device, "s2io", &[(0, Rw), (1, Ro), (2, UniDyn)]);
        let s1_bgl = bgl(&gpu.device, "s1", &[(0, Rw), (1, UniDyn)]);
        let ah_bgl = bgl(
            &gpu.device,
            "ah",
            &[(0, Ro), (1, Ro), (2, Ro), (3, Rw), (4, Rw), (5, UniDyn)],
        );
        let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni"),
            size: UNI_SLOT * UNI_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = |bgl: &wgpu::BindGroupLayout| {
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(bgl)],
                    ..Default::default()
                })
        };
        let pipe = |label: &str,
                    src: &str,
                    entry: &str,
                    ly: &wgpu::PipelineLayout|
         -> Result<wgpu::ComputePipeline> {
            gpu.pipeline(label, src, entry, Some(ly))
        };
        crate::browser::log("[enc] pipelines");
        let gemm = pipe(
            "gemm",
            &shaders::prefill_gemm(false, false, false, false, false),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_av = pipe(
            "gemm_av",
            &shaders::prefill_gemm(true, false, false, false, false),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_bias = pipe(
            "gemm_bias",
            &shaders::prefill_gemm(false, false, true, false, false),
            "gemm",
            &layout(&gemm_bias_bgl),
        )?;
        let ln_src = format!(
            "{}
{}",
            shaders::HALF_AT,
            shaders::LAYER_NORM
        );
        let ln_f32_src = P_LN_F32;
        let ln = pipe("ln", &ln_src, "layernorm", &layout(&ln_bgl))?;
        let ln_f32 = pipe("ln_f32", ln_f32_src, "main", &layout(&ln_bgl))?;
        let silu_f32 = pipe("silu_f32", P_SILU_F32, "main", &layout(&s1_bgl))?;
        let glu_f32 = pipe("glu_f32", P_GLU_F32, "main", &layout(&s2_bgl))?;
        let add_scaled_f32 = pipe(
            "add_scaled_f32",
            P_ADD_SCALED_F32,
            "main",
            &layout(&s2io_bgl),
        )?;
        let f2h_mix = pipe("f2h_mix", P_F2H_MIX, "main", &layout(&s2_bgl))?;
        let add_head_f32b = pipe("add_head_f32b", P_ADD_HEAD_F32B, "main", &layout(&ah_bgl))?;
        let h2f_f32b = pipe("h2f_f32b", P_H2F_F32B, "main", &layout(&s2_bgl))?;
        let dwconv_f32 = pipe("dwconv_f32", P_DWCONV9_F32, "main", &layout(&u4_bgl))?;
        let add_bias_f32 = pipe("add_bias_f32", P_ADD_BIAS_F32, "main", &layout(&s2io_bgl))?;
        let bn_silu_f32 = pipe("bn_silu_f32", P_BN_SILU_F32, "main", &layout(&u4_bgl))?;
        let silu = pipe("silu", shaders::SILU_INPLACE, "main", &layout(&s1_bgl))?;
        let add = pipe("add", shaders::ADD, "main", &layout(&s2io_bgl))?;
        let add_scaled = pipe("add_scaled", P_ADD_SCALED, "main", &layout(&s2io_bgl))?;
        let gemm_f32 = pipe("gemm_f32", &gemm_mix(false), "gemm", &layout(&gemm_bgl))?;
        let gemm_f32_tb = pipe("gemm_f32_tb", &gemm_mix(true), "gemm", &layout(&gemm_bgl))?;
        let gemm_lb_bm64 = pipe(
            "gemm_lb_bm64",
            &gemm_mix_bm64_lb(),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_bm64 = pipe(
            "gemm_f32_bm64",
            &gemm_mix_bm64(false),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_tb_bm64 = pipe(
            "gemm_f32_tb_bm64",
            &gemm_mix_bm64(true),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_4x8 = pipe(
            "gemm_f32_4x8",
            &gemm_mix_4x8(false),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_tb_4x8 = pipe(
            "gemm_f32_tb_4x8",
            &gemm_mix_4x8(true),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_8x4 = pipe(
            "gemm_f32_8x4",
            &gemm_mix_8x4(false),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let gemm_f32_tb_8x4 = pipe(
            "gemm_f32_tb_8x4",
            &gemm_mix_8x4(true),
            "gemm",
            &layout(&gemm_bgl),
        )?;
        let add_head_f32 = pipe("add_head_f32", P_ADD_HEAD_F32, "main", &layout(&ah_bgl))?;
        let rel_add_f32 = pipe("rel_add_f32", P_REL_ADD_F32, "main", &layout(&u4_bgl))?;
        let add_softmax_f32 = pipe(
            "add_softmax_f32",
            P_ADD_SOFTMAX_F32,
            "main",
            &layout(&u4_bgl),
        )?;
        let softmax_f32 = pipe("softmax_f32", P_SOFTMAX_F32, "main", &layout(&s2_bgl))?;
        let h2f_f32 = pipe("h2f_f32", P_H2F_F32, "main", &layout(&s2_bgl))?;
        let add_bias = pipe("add_bias", shaders::ADD_BIAS, "main", &layout(&s2io_bgl))?;
        let glu = pipe("glu", P_GLU, "main", &layout(&s2_bgl))?;
        let add_head = pipe("add_head", P_ADD_HEAD, "main", &layout(&ah_bgl))?;
        let f2h = pipe("f2h", P_FLAT_TO_HEADS, "main", &layout(&s2_bgl))?;
        let h2f = pipe("h2f", P_HEADS_TO_FLAT, "main", &layout(&s2_bgl))?;
        let rel_add = pipe("rel_add", P_REL_ADD, "main", &layout(&u4_bgl))?;
        let softmax = pipe("p_softmax", P_SOFTMAX, "main", &layout(&s2_bgl))?;
        let dwconv = pipe("dwconv9", P_DWCONV9, "main", &layout(&u4_bgl))?;
        let bn_silu = pipe("bn_silu", P_BN_SILU, "main", &layout(&u4_bgl))?;

        let gemm_av_bgl = gemm_bgl.clone();
        crate::browser::log("[enc] struct ready");
        Ok(Self {
            w: EncoderWeights {
                sub_w,
                sub_b,
                layers,
                proj_w,
                proj_b,
            },
            gemm,
            gemm_av,
            gemm_bias,
            ln,
            silu,
            add,
            add_scaled,
            gemm_f32,
            gemm_f32_tb,
            gemm_f32_bm64,
            gemm_lb_bm64,
            gemm_f32_tb_bm64,
            gemm_f32_4x8,
            gemm_f32_tb_4x8,
            gemm_f32_8x4,
            gemm_f32_tb_8x4,
            ln_f32,
            silu_f32,
            glu_f32,
            add_scaled_f32,
            f2h_mix,
            add_head_f32b,
            add_bias_f32,
            h2f_f32b,
            dwconv_f32,
            bn_silu_f32,
            add_head_f32,
            rel_add_f32,
            softmax_f32,
            add_softmax_f32,
            h2f_f32,
            add_bias,
            glu,
            add_head,
            f2h,
            h2f,
            rel_add,
            softmax,
            dwconv,
            bn_silu,
            gemm_bgl,
            gemm_av_bgl,
            gemm_bias_bgl,
            ln_bgl,
            u4_bgl,
            s2_bgl,
            s2io_bgl,
            s1_bgl,
            ah_bgl,
            uni,
            prof: if std::env::var("BARK3_ENC_PROFILE").is_ok() {
                Prof::new(&gpu, "enc_profile")
            } else {
                None
            },
            gpu,
        })
    }
}

// ───────────────────────────── encode ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U6 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    f: u32,
}

pub struct Ctx<'a> {
    pub e: &'a Encoder,
    pub slot: u32,
}

impl<'a> Ctx<'a> {
    pub fn new(e: &'a Encoder, slot: u32) -> Self {
        Self { e, slot }
    }

    /// Records an operator boundary for `BARK3_ENC_PROFILE`; a no-op otherwise.
    fn mark(&self, cp: &mut wgpu::ComputePass<'_>, label: &'static str) {
        if let Some(prof) = &self.e.prof {
            prof.mark(cp, label);
        }
    }

    pub fn next_slot(&mut self, bytes: &[u8]) -> u32 {
        let s = self.slot;
        self.e
            .gpu
            .queue
            .write_buffer(&self.e.uni, s as u64 * UNI_SLOT, bytes);
        self.slot += 1;
        assert!(self.slot < UNI_SLOTS as u32, "uniform slots exhausted");
        s
    }

    /// bind group: explicit (binding, buffer, byte-offset) entries + dynamic uniform at `uni_binding`.
    fn bg(
        &self,
        bgl: &wgpu::BindGroupLayout,
        entries: &[(u32, &wgpu::Buffer, u64)],
        uni_binding: u32,
        slot: u32,
    ) -> wgpu::BindGroup {
        let mut bg_entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(b, buf, off)| wgpu::BindGroupEntry {
                binding: *b,
                resource: if *off == 0 {
                    buf.as_entire_binding()
                } else {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: *off,
                        size: None,
                    })
                },
            })
            .collect();
        let _ = slot;
        bg_entries.push(wgpu::BindGroupEntry {
            binding: uni_binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.e.uni,
                offset: 0,
                size: Some(std::num::NonZeroU64::new(64).unwrap()),
            }),
        });
        self.e
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: &bg_entries,
            })
    }

    pub fn gemm(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let (pipe, bgl) = if transb {
            (&self.e.gemm_av, &self.e.gemm_av_bgl)
        } else {
            (&self.e.gemm, &self.e.gemm_bgl)
        };
        let bg = self.bg(bgl, &[(0, a, 0), (1, w, 0), (2, c, 0)], 3, s);
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 128, z);
    }

    pub fn gemm_bias(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa: 0,
            bsb: 0,
            bsc: 0,
            beta: 0,
            row0: 0,
            lda: k,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.gemm_bias_bgl,
            &[(0, a, 0), (1, w, 0), (2, c, 0), (4, bias, 0)],
            3,
            s,
        );
        cp.set_pipeline(&self.e.gemm_bias);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 128, 1);
    }

    pub fn ln(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        src: &wgpu::Buffer,
        wb: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        outer: u32,
    ) {
        // LnCfg { last, outer, eps, _pad }
        let bytes: [u32; 4] = [1024, outer, cfg::LN_EPS.to_bits(), 0];
        let s = self.next_slot(bytemuck::cast_slice(&bytes));
        let bg = self.bg(
            &self.e.ln_bgl,
            &[(0, src, 0), (1, wb, 0), (2, wb, 1024 * 2), (4, dst, 0)],
            3,
            s,
        );
        cp.set_pipeline(&self.e.ln);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(outer, 1, 1);
    }

    pub fn silu(&mut self, cp: &mut wgpu::ComputePass<'_>, x: &wgpu::Buffer, n2: u32) {
        let s = self.next_slot(bytemuck::cast_slice(&[n2, 0, 0, 0]));
        let bg = self.bg(&self.e.s1_bgl, &[(0, x, 0)], 1, s);
        cp.set_pipeline(&self.e.silu);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n2.div_ceil(256), 1, 1);
    }

    /// Mixed GEMM: A (m,k) f32 · W -> C (m,n) f32 (z batches); transb: W is [k, n].
    ///
    /// `w` carries its own layout: f16 halves for the bf16 checkpoint, or
    /// 4-bit codes plus a level table for a low-bit checkpoint (see [`LowBit`]).
    /// The accumulation order is identical in both cases, so a weight tensor
    /// gives the same result whichever way it was stored.
    pub fn gemm_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &Weight,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let lb = w.lowbit();
        assert!(!(transb && lb.is_some()), "low-bit weights are A*W only");
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: lb.map_or(0, |m| m.lev_words),
            k_lev: lb.map_or(0, |m| m.k_lev),
            codes_ld: lb.map_or(0, |m| m.codes_ld),
            lowbit: u32::from(lb.is_some()),
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.gemm_bgl,
            &[(0, a, 0), (1, &w.buf, 0), (2, c, 0)],
            3,
            s,
        );
        // 64x128 tile (128 threads): higher occupancy than the 128x128 kernel.
        self.mark(
            cp,
            if lb.is_some() {
                "gemm_lb"
            } else {
                gemm_label(n, k)
            },
        );
        cp.set_pipeline(if lb.is_some() {
            &self.e.gemm_lb_bm64
        } else if transb {
            &self.e.gemm_f32_tb_bm64
        } else {
            &self.e.gemm_f32_bm64
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 64, z);
    }

    /// 128x128 tile / 256 threads, retained as an A/B against the 64x128 kernel.
    pub fn gemm_f32_128(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.gemm_bgl, &[(0, a, 0), (1, w, 0), (2, c, 0)], 3, s);
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb
        } else {
            &self.e.gemm_f32
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 128, z);
    }

    pub fn gemm_f32_4x8(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.gemm_bgl, &[(0, a, 0), (1, w, 0), (2, c, 0)], 3, s);
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb_4x8
        } else {
            &self.e.gemm_f32_4x8
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 64, z);
    }

    /// gemm_f32 with raw byte offsets into A/W/C (all must be 256-aligned).
    pub fn gemm_f32_off(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        a_off: u64,
        w_off: u64,
        c_off: u64,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.gemm_bgl,
            &[(0, a, a_off), (1, w, w_off), (2, c, c_off)],
            3,
            s,
        );
        self.mark(cp, "gemm_off_bm64");
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb_bm64
        } else {
            &self.e.gemm_f32_bm64
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 64, z);
    }

    /// 64x64 kernel with raw byte offsets — used for the batched attention
    /// score matrices, where its higher occupancy wins over 64x128.
    pub fn gemm_f32_off_8x4(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        a_off: u64,
        w_off: u64,
        c_off: u64,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.gemm_bgl,
            &[(0, a, a_off), (1, w, w_off), (2, c, c_off)],
            3,
            s,
        );
        self.mark(cp, "gemm_score_off");
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb_8x4
        } else {
            &self.e.gemm_f32_8x4
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 64, m / 64, z);
    }

    pub fn gemm_f32_bm64(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.gemm_bgl, &[(0, a, 0), (1, w, 0), (2, c, 0)], 3, s);
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb_bm64
        } else {
            &self.e.gemm_f32_bm64
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 128, m / 64, z);
    }

    /// 64x64 tile / 128 threads / 8x4 per thread: experimental alternative to
    /// the production 64x128 kernel, opt-in via `GB_8X4`.  n and m must be
    /// multiples of 64.
    pub fn gemm_f32_8x4(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        transb: bool,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        n: u32,
        k: u32,
        ldc: u32,
        bsa: u32,
        bsb: u32,
        bsc: u32,
        lda: u32,
        z: u32,
    ) {
        let d = GDims {
            m,
            n,
            k,
            ldc,
            bsa,
            bsb,
            bsc,
            beta: 0,
            row0: 0,
            lda,
            _pad: 0,
            lev_words: 0,
            k_lev: 0,
            codes_ld: 0,
            lowbit: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.gemm_bgl, &[(0, a, 0), (1, w, 0), (2, c, 0)], 3, s);
        self.mark(cp, "gemm_score");
        cp.set_pipeline(if transb {
            &self.e.gemm_f32_tb_8x4
        } else {
            &self.e.gemm_f32_8x4
        });
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n / 64, m / 64, z);
    }

    pub fn ln_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        src: &wgpu::Buffer,
        wb: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        outer: u32,
    ) {
        let bytes: [u32; 4] = [1024, outer, 1e-5f32.to_bits(), 0];
        let s = self.next_slot(bytemuck::cast_slice(&bytes));
        let bg = self.bg(
            &self.e.ln_bgl,
            &[(0, src, 0), (1, wb, 0), (2, wb, 4096), (4, dst, 0)],
            3,
            s,
        );
        self.mark(cp, "ln");
        cp.set_pipeline(&self.e.ln_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(outer, 1, 1);
    }

    pub fn silu_f32(&mut self, cp: &mut wgpu::ComputePass<'_>, x: &wgpu::Buffer, n: u32) {
        let s = self.next_slot(bytemuck::cast_slice(&[n, 0, 0, 0]));
        let bg = self.bg(&self.e.s1_bgl, &[(0, x, 0)], 1, s);
        self.mark(cp, "silu");
        cp.set_pipeline(&self.e.silu_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    pub fn glu_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        c: u32,
        total: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[c, total, 0, 0]));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        self.mark(cp, "glu");
        cp.set_pipeline(&self.e.glu_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub fn add_scaled_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        scale: f32,
        n: u32,
    ) {
        let bytes: [u32; 4] = [n, scale.to_bits(), 0, 0];
        let s = self.next_slot(bytemuck::cast_slice(&bytes));
        let bg = self.bg(&self.e.s2io_bgl, &[(0, a, 0), (1, b, 0)], 2, s);
        self.mark(cp, "add_scaled");
        cp.set_pipeline(&self.e.add_scaled_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    pub fn f2h_mix(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
        stride: u32,
        coff: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: stride,
            d: coff,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        self.mark(cp, "f2h_mix");
        cp.set_pipeline(&self.e.f2h_mix);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn add_head_f32b(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        q: &wgpu::Buffer,
        u: &wgpu::Buffer,
        v: &wgpu::Buffer,
        qu: &wgpu::Buffer,
        qv: &wgpu::Buffer,
        t: u32,
        words: u32,
        stride: u32,
        coff: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: stride,
            d: coff,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.ah_bgl,
            &[(0, q, 0), (1, u, 0), (2, v, 0), (3, qu, 0), (4, qv, 0)],
            5,
            s,
        );
        self.mark(cp, "add_head_f32b");
        cp.set_pipeline(&self.e.add_head_f32b);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn h2f_f32b(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        self.mark(cp, "h2f_f32b");
        cp.set_pipeline(&self.e.h2f_f32b);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn dwconv_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[t, words, 0, 0]));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, w, 0), (2, out, 0)], 3, s);
        self.mark(cp, "dwconv");
        cp.set_pipeline(&self.e.dwconv_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn bn_silu_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        ab: &wgpu::Buffer,
        out: &wgpu::Buffer,
        words: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[0u32, words, 0, 0]));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, ab, 0), (2, out, 0)], 3, s);
        self.mark(cp, "bn_silu");
        cp.set_pipeline(&self.e.bn_silu_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups((words * 2).div_ceil(256), 1, 1);
    }

    pub fn add_bias_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        b: &wgpu::Buffer,
        outer: u32,
        last: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[outer, last, 0, 0]));
        let bg = self.bg(&self.e.s2io_bgl, &[(0, x, 0), (1, b, 0)], 2, s);
        self.mark(cp, "add_bias");
        cp.set_pipeline(&self.e.add_bias_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        let total = outer * last;
        cp.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub fn rel_add_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        bd: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t_real: u32,
        tp: u32,
        pp: u32,
        scale: f32,
    ) {
        let d = U7 {
            a: t_real,
            b: tp,
            c: pp,
            d: scale,
            e: 0,
            f: 0,
            g: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, bd, 0), (2, out, 0)], 3, s);
        self.mark(cp, "rel_add");
        cp.set_pipeline(&self.e.rel_add_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        let words = 8 * tp * tp;
        let groups = words.div_ceil(256);
        if groups <= 65535 {
            cp.dispatch_workgroups(groups, 1, 1);
        } else {
            cp.dispatch_workgroups(65535, groups.div_ceil(65535), 1);
        }
    }

    /// `scale * (content + bd)` followed by a row softmax, in one dispatch.
    ///
    /// One workgroup owns one `(head, frame)` row: it reads the content score and
    /// the relative-position window once, keeps the biased row in workgroup
    /// memory, and writes the probabilities once. The bias, the max and the sum
    /// are evaluated in the same order as [`Self::rel_add_f32`] +
    /// [`Self::softmax_f32`], so `out` is bit-identical to that pair (and the
    /// dump/stage checks keep matching).
    ///
    /// Rows longer than [`MAX_ADD_SOFTMAX_ROW`] fall back to the two kernels.
    pub fn add_softmax_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        content: &wgpu::Buffer,
        bd: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t_real: u32,
        tp: u32,
        pp: u32,
        scale: f32,
    ) {
        debug_assert!(t_real as usize <= MAX_ADD_SOFTMAX_ROW);
        let d = U7 {
            a: t_real,
            b: tp,
            c: pp,
            d: scale,
            e: 0,
            f: 0,
            g: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.u4_bgl,
            &[(0, content, 0), (1, bd, 0), (2, out, 0)],
            3,
            s,
        );
        self.mark(cp, "add_softmax");
        cp.set_pipeline(&self.e.add_softmax_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        // One workgroup per (head, frame) row; 65535 groups per dimension is the
        // device limit, so the head/frame grid is flattened into two.
        let rows = 8 * tp;
        cp.dispatch_workgroups(rows.min(65535), rows.div_ceil(65535), 1);
    }

    pub fn softmax_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: u32,
        n: u32,
        stride: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[rows, n, stride, 0]));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        self.mark(cp, "softmax");
        cp.set_pipeline(&self.e.softmax_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(rows, 1, 1);
    }

    pub fn h2f_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        cp.set_pipeline(&self.e.h2f_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn add_scaled(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        scale: f32,
        n2: u32,
    ) {
        let bytes: [u32; 4] = [n2, scale.to_bits(), 0, 0];
        let s = self.next_slot(bytemuck::cast_slice(&bytes));
        let bg = self.bg(&self.e.s2io_bgl, &[(0, a, 0), (1, b, 0)], 2, s);
        cp.set_pipeline(&self.e.add_scaled);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n2.div_ceil(256), 1, 1);
    }

    pub fn add(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n2: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[n2, 0, 0, 0]));
        let bg = self.bg(&self.e.s2io_bgl, &[(0, a, 0), (1, b, 0)], 2, s);
        cp.set_pipeline(&self.e.add);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(n2.div_ceil(256), 1, 1);
    }

    pub fn add_bias(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        b: &wgpu::Buffer,
        outer: u32,
        last: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[outer, last, 0, 0]));
        let bg = self.bg(&self.e.s2io_bgl, &[(0, x, 0), (1, b, 0)], 2, s);
        cp.set_pipeline(&self.e.add_bias);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        let total2 = outer * (last / 2);
        cp.dispatch_workgroups(total2.div_ceil(256), 1, 1);
    }

    pub fn glu(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        c2: u32,
        total2: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[c2, total2, 0, 0]));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        cp.set_pipeline(&self.e.glu);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(total2.div_ceil(256), 1, 1);
    }

    pub fn add_head_f32(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        q: &wgpu::Buffer,
        u: &wgpu::Buffer,
        v: &wgpu::Buffer,
        qu: &wgpu::Buffer,
        qv: &wgpu::Buffer,
        t: u32,
        words: u32,
        stride: u32,
        coff: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: stride / 2,
            d: coff / 2,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.ah_bgl,
            &[(0, q, 0), (1, u, 0), (2, v, 0), (3, qu, 0), (4, qv, 0)],
            5,
            s,
        );
        cp.set_pipeline(&self.e.add_head_f32);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn add_head(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        q: &wgpu::Buffer,
        u: &wgpu::Buffer,
        v: &wgpu::Buffer,
        qu: &wgpu::Buffer,
        qv: &wgpu::Buffer,
        t: u32,
        words: u32,
        stride: u32,
        coff: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: stride / 2,
            d: coff / 2,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(
            &self.e.ah_bgl,
            &[(0, q, 0), (1, u, 0), (2, v, 0), (3, qu, 0), (4, qv, 0)],
            5,
            s,
        );
        cp.set_pipeline(&self.e.add_head);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn f2h(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
        stride: u32,
        coff: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: stride / 2,
            d: coff / 2,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        cp.set_pipeline(&self.e.f2h);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn h2f(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
    ) {
        let d = U6 {
            a: t,
            b: words,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        cp.set_pipeline(&self.e.h2f);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn rel_add(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        bd: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t_real: u32,
        tp: u32,
        pp: u32,
        scale: f32,
    ) {
        let d = U7 {
            a: t_real,
            b: tp,
            c: pp,
            d: scale,
            e: 0,
            f: 0,
            g: 0,
        };
        let s = self.next_slot(bytemuck::bytes_of(&d));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, bd, 0), (2, out, 0)], 3, s);
        cp.set_pipeline(&self.e.rel_add);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        let words = 8 * tp * tp / 2;
        let groups = words.div_ceil(256);
        if groups <= 65535 {
            cp.dispatch_workgroups(groups, 1, 1);
        } else {
            let gy = groups.div_ceil(65535);
            cp.dispatch_workgroups(65535, gy, 1);
        }
    }

    pub fn softmax(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: u32,
        n: u32,
        stride: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[rows, n, stride, 0]));
        let bg = self.bg(&self.e.s2_bgl, &[(0, x, 0), (1, out, 0)], 2, s);
        cp.set_pipeline(&self.e.softmax);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(rows, 1, 1);
    }

    pub fn dwconv(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        out: &wgpu::Buffer,
        t: u32,
        words: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[t, words, 0, 0]));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, w, 0), (2, out, 0)], 3, s);
        cp.set_pipeline(&self.e.dwconv);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }

    pub fn bn_silu(
        &mut self,
        cp: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        ab: &wgpu::Buffer,
        out: &wgpu::Buffer,
        words: u32,
    ) {
        let s = self.next_slot(bytemuck::cast_slice(&[0u32, words, 0, 0]));
        let bg = self.bg(&self.e.u4_bgl, &[(0, x, 0), (1, ab, 0), (2, out, 0)], 3, s);
        cp.set_pipeline(&self.e.bn_silu);
        cp.set_bind_group(0, &bg, &[s * UNI_SLOT as u32]);
        cp.dispatch_workgroups(words.div_ceil(256), 1, 1);
    }
}

pub struct EncodeOut {
    /// (t_real, 640) encoder projector output, f32
    pub pooler: Vec<f32>,
    /// same data still resident on the GPU (for the GPU TDT decoder)
    pub pooler_buf: Option<wgpu::Buffer>,
    /// optional dump of block `dump_layer` output (t_real, 1024) f32
    pub block: Option<Vec<f32>>,
}

impl Encoder {
    /// `sub_rows`: (t_real, 4096) conv-subsampled features (CPU path).
    pub fn encode(
        &self,
        sub_rows: &[f32],
        t_real: usize,
        dump_layer: Option<usize>,
    ) -> Result<EncodeOut> {
        let tp = pad128(t_real);
        let mut sub_pad = vec![0f32; tp * 4096];
        for (i, v) in sub_rows.iter().enumerate() {
            sub_pad[i] = *v;
        }
        let sub = self.gpu.storage("a_sub", (tp * 4096 * 4) as u64);
        self.gpu
            .queue
            .write_buffer(&sub, 0, bytemuck::cast_slice(&sub_pad));
        self.encode_pre(&sub, t_real, dump_layer, true)
    }

    /// Same as [Self::encode] but consumes a GPU buffer that already holds the
    /// (pad128(t_real), 4096) subsampled rows (GPU front end).
    pub fn encode_pre(
        &self,
        sub: &wgpu::Buffer,
        t_real: usize,
        dump_layer: Option<usize>,
        read_pooler: bool,
    ) -> Result<EncodeOut> {
        let tp = pad128(t_real) as u32;
        let t_real = t_real as u32;
        let p_real = 2 * t_real - 1;
        let pp = pad128(p_real as usize) as u32;
        let scale: f32 = (cfg::HEAD_DIM as f32).powf(-0.5);
        // Rows beyond t_real are never read back and cannot influence real rows
        // (attention keys are masked to t_real), so only dispatch whole 64-row
        // blocks that contain real frames.
        let te = ((t_real as usize).div_ceil(64) * 64) as u32;

        let mut pos_pad = vec![0f32; pp as usize * 1024];
        let pos = pos_embed_f32(t_real as usize);
        pos_pad[..pos.len()].copy_from_slice(&pos);

        let g = &self.gpu;
        let mk = |label: &str, words: u64| g.storage(label, words * 4);
        let h = mk("a_h", tp as u64 * 1024);
        let h2 = mk("a_h2", tp as u64 * 1024);
        let tmp = mk("a_tmp", tp as u64 * 1024);
        let normed = mk("a_normed", tp as u64 * 1024);
        let normed3 = mk("a_normed3", tp as u64 * 1024);
        let tmp2 = mk("a_tmp2", tp as u64 * 1024);
        let qkv = mk("a_qkv", tp as u64 * 3072);
        let qu = mk("a_qu", tp as u64 * 1024);
        let qv = mk("a_qv", tp as u64 * 1024);
        let kh = mk("a_kh", tp as u64 * 512);
        let vh = mk("a_vh", tp as u64 * 512);
        let relk_flat = mk("a_rk", pp as u64 * 1024);
        let rkh = mk("a_rkh", pp as u64 * 512);
        let content = mk("a_sc", tp as u64 * tp as u64 * 8);
        let attn = mk("a_attn", tp as u64 * tp as u64 * 8);
        let bd = mk("a_bd", tp as u64 * pp as u64 * 8);
        let attn_h = mk("a_ah", tp as u64 * 1024);
        let attnout = mk("a_ao", tp as u64 * 1024);
        let ffb = mk("a_ffb", tp as u64 * 4096);
        let cb1 = mk("a_cb1", tp as u64 * 2048);
        let cb2 = mk("a_cb2", tp as u64 * 1024);
        let pooler = mk("a_pool", tp as u64 * 640);
        let pos = mk("a_pos", pp as u64 * 1024);
        g.queue
            .write_buffer(&pos, 0, bytemuck::cast_slice(&pos_pad));

        let mut block_dump: Option<Vec<f32>> = None;
        let mut ctx = Ctx { e: self, slot: 0 };
        let mut enc = g.device.create_command_encoder(&Default::default());
        let mut layers_in_chunk = 0u32;

        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            ctx.gemm_f32(
                &mut cp,
                false,
                sub,
                &self.w.sub_w,
                &h,
                te,
                1024,
                4096,
                1024,
                0,
                0,
                0,
                4096,
                1,
            );
            ctx.add_bias_f32(&mut cp, &h, &self.w.sub_b, te, 1024);
        }
        if std::env::var("BARK3_STAGE").as_deref() == Ok("sublin") {
            g.queue.submit([enc.finish()]);
            g.wait();
            let bytes = g.readback(&h, t_real as u64 * 1024 * 4)?;
            let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            return Ok(EncodeOut {
                pooler: vec![],
                pooler_buf: None,
                block: Some(out),
            });
        }

        let mut h_cur = h.clone();
        let mut h_alt = h2.clone();
        crate::browser::log(&format!("[enc] forward tp={tp} te={te} pp={pp}"));
        let lstart = crate::clock::now();
        for l in 0..cfg::LAYERS {
            if l % 2 == 0 {
                crate::browser::log(&format!(
                    "[enc] layer {l} at {:.1}s",
                    lstart.elapsed().as_secs_f64()
                ));
            }
            let lw = &self.w.layers[l];
            let w512 = tp * 512; // words per (T,1024)
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                // ── feed-forward half 1: h += 0.5 * ff2(silu(ff1(ln1(h))))
                ctx.ln_f32(&mut cp, &h_cur, &lw.ln[0], &normed, te);
                ctx.gemm_f32(
                    &mut cp, false, &normed, &lw.ff1a, &ffb, te, 4096, 1024, 4096, 0, 0, 0, 1024, 1,
                );
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ffb1") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("ffb".into(), &ffb, tp as u64 * 4096),
                            ("normed".into(), &normed, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.silu_f32(&mut cp, &ffb, te * 4096);
                ctx.gemm_f32(
                    &mut cp, false, &ffb, &lw.ff1b, &tmp, te, 1024, 4096, 1024, 0, 0, 0, 4096, 1,
                );
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ff1bout") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("tmp".into(), &tmp, tp as u64 * 1024),
                            ("ffb".into(), &ffb, tp as u64 * 4096),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.add_scaled_f32(&mut cp, &h_cur, &tmp, 0.5, te * 1024);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ff1") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(enc, &[("h".into(), &h_cur, tp as u64 * 1024)])
                    {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                // ── attention
                ctx.ln_f32(&mut cp, &h_cur, &lw.ln[1], &normed, te);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ln1") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("normed".into(), &normed, tp as u64 * 1024),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.gemm_f32(
                    &mut cp, false, &normed, &lw.qkv, &qkv, te, 3072, 1024, 3072, 0, 0, 0, 1024, 1,
                );
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("qkv3") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("qkv".into(), &qkv, tp as u64 * 3072),
                            ("normed".into(), &normed, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("qkv2") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("qkv2".into(), &qkv, tp as u64 * 1536),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.add_head_f32b(&mut cp, &qkv, &lw.bu, &lw.bv, &qu, &qv, tp, w512, 3072, 0);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("qu") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("qu".into(), &qu, tp as u64 * 1024),
                            ("qkv".into(), &qkv, tp as u64 * 3072),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.f2h_mix(&mut cp, &qkv, &kh, tp, w512, 3072, 1024);
                ctx.f2h_mix(&mut cp, &qkv, &vh, tp, w512, 3072, 2048);
                ctx.gemm_f32(
                    &mut cp, false, &pos, &lw.relk, &relk_flat, pp, 1024, 1024, 1024, 0, 0, 0,
                    1024, 1,
                );
                ctx.f2h_mix(&mut cp, &relk_flat, &rkh, pp, pp * 512, 1024, 0);
                // k=128, z=8: the 64x64/8x4 kernel is ~16% faster here.
                ctx.gemm_f32_8x4(
                    &mut cp,
                    false,
                    &qu,
                    &kh,
                    &content,
                    te,
                    tp,
                    128,
                    tp,
                    tp * 128,
                    tp * 64,
                    tp * tp,
                    128,
                    8,
                );
                {
                    let t = t_real as usize;
                    for ib in 0..t.div_ceil(128) {
                        let i0 = ib * 128;
                        let i_max = (i0 + 127).min(t - 1);
                        let jmin = t - 1 - i_max;
                        let jmax = (2 * t - 2 - i0).min(pp as usize - 1);
                        let j0 = (jmin / 128) * 128;
                        let nwin = ((jmax - j0 + 1).div_ceil(128) * 128).min(pp as usize - j0);
                        let a_off = (i0 as u64) * 128 * 4;
                        let w_off = (j0 as u64) * 64 * 4;
                        let c_off = ((i0 as u64) * pp as u64 + j0 as u64) * 4;
                        ctx.gemm_f32_off_8x4(
                            &mut cp,
                            false,
                            &qv,
                            &rkh,
                            &bd,
                            a_off,
                            w_off,
                            c_off,
                            128,
                            nwin as u32,
                            128,
                            pp,
                            tp * 128,
                            pp * 64,
                            tp * pp,
                            128,
                            8,
                        );
                    }
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("cbd") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("cbd".into(), &bd, tp as u64 * pp as u64 * 4),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                // Scores -> probabilities. One fused kernel moves the (8, T, T)
                // matrix twice instead of five times; the two-kernel sequence
                // stays for rows too long to cache (and as the A/B reference).
                let fused = t_real as usize <= MAX_ADD_SOFTMAX_ROW
                    && std::env::var("BARK3_SOFTMAX2").is_err();
                let probs = if fused {
                    ctx.add_softmax_f32(&mut cp, &content, &bd, &attn, t_real, tp, pp, scale);
                    &attn
                } else {
                    ctx.rel_add_f32(&mut cp, &content, &bd, &attn, t_real, tp, pp, scale);
                    ctx.softmax_f32(&mut cp, &attn, &content, 8 * tp, t_real, tp);
                    &content
                };
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("probs") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("probs".into(), probs, tp as u64 * tp as u64 * 8),
                            ("vh".into(), &vh, tp as u64 * 256),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.gemm_f32(
                    &mut cp,
                    true,
                    probs,
                    &Weight::f16(vh.clone()),
                    &attn_h,
                    te,
                    128,
                    tp,
                    128,
                    tp * tp,
                    tp * 64,
                    tp * 128,
                    tp,
                    8,
                );
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("cav") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("cav".into(), &attn_h, tp as u64 * 512),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.h2f_f32b(&mut cp, &attn_h, &attnout, tp, w512);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("h2f2") {
                    drop(cp);
                    if let Some(o) =
                        self.stage_exit(enc, &[("attnout".into(), &attnout, tp as u64 * 1024)])
                    {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("h2f") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("attnout".into(), &attnout, tp as u64 * 1024),
                            ("attn_h".into(), &attn_h, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ch2f") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("ch2f".into(), &attnout, tp as u64 * 512),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.gemm_f32(
                    &mut cp, false, &attnout, &lw.o, &tmp, te, 1024, 1024, 1024, 0, 0, 0, 1024, 1,
                );
                ctx.add_scaled_f32(&mut cp, &h_cur, &tmp, 1.0, te * 1024);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("attn2") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("h".into(), &h_cur, tp as u64 * 1024),
                            ("tmp".into(), &tmp, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("attn") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(enc, &[("h".into(), &h_cur, tp as u64 * 1024)])
                    {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                // ── conv module
                let no_conv = std::env::var("BARK3_NOCONV").is_ok();
                let no_dw = std::env::var("BARK3_NODW").is_ok();
                if !no_conv {
                    let ln2_w: &wgpu::Buffer = if std::env::var("BARK3_LN2SWAP").is_ok() {
                        &lw.ln[0]
                    } else {
                        &lw.ln[2]
                    };
                    let ln2_dst: &wgpu::Buffer = if std::env::var("BARK3_LN2FRESH").is_ok() {
                        &normed3
                    } else {
                        &normed
                    };
                    ctx.ln_f32(&mut cp, &h_cur, ln2_w, ln2_dst, te);
                    if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ln2") {
                        drop(cp);
                        if let Some(o) = self.stage_exit(
                            enc,
                            &[
                                ("normed".into(), ln2_dst, tp as u64 * 1024),
                                ("h".into(), &h_cur, tp as u64 * 1024),
                            ],
                        ) {
                            return Ok(o);
                        }
                        enc = g.device.create_command_encoder(&Default::default());
                        ctx.slot = 0;
                        let cp2 = enc.begin_compute_pass(&Default::default());
                        cp = cp2;
                    }
                    ctx.gemm_f32(
                        &mut cp, false, &normed, &lw.pw1, &cb1, te, 2048, 1024, 2048, 0, 0, 0,
                        1024, 1,
                    );
                    if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("pw1") {
                        drop(cp);
                        if let Some(o) = self.stage_exit(
                            enc,
                            &[
                                ("cb1".into(), &cb1, tp as u64 * 2048),
                                ("normed".into(), &normed, tp as u64 * 1024),
                            ],
                        ) {
                            return Ok(o);
                        }
                        enc = g.device.create_command_encoder(&Default::default());
                        ctx.slot = 0;
                        let cp2 = enc.begin_compute_pass(&Default::default());
                        cp = cp2;
                    }
                    ctx.glu_f32(&mut cp, &cb1, &cb2, 1024, te * 1024);
                    if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("glu") {
                        drop(cp);
                        if let Some(o) = self.stage_exit(
                            enc,
                            &[
                                ("cb2".into(), &cb2, tp as u64 * 1024),
                                ("cb1".into(), &cb1, tp as u64 * 2048),
                            ],
                        ) {
                            return Ok(o);
                        }
                        enc = g.device.create_command_encoder(&Default::default());
                        ctx.slot = 0;
                        let cp2 = enc.begin_compute_pass(&Default::default());
                        cp = cp2;
                    }
                    if no_dw {
                        // bypass dw+bn: copy cb2 through tmp2 is not needed; feed cb2 directly
                        ctx.gemm_f32(
                            &mut cp, false, &cb2, &lw.pw2, &tmp, tp, 1024, 1024, 1024, 0, 0, 0,
                            1024, 1,
                        );
                    } else {
                        ctx.dwconv_f32(&mut cp, &cb2, &lw.dw, &tmp2, t_real, tp * 512);
                        if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("dw") {
                            drop(cp);
                            if let Some(o) = self.stage_exit(
                                enc,
                                &[
                                    ("tmp2".into(), &tmp2, tp as u64 * 1024),
                                    ("cb2".into(), &cb2, tp as u64 * 1024),
                                ],
                            ) {
                                return Ok(o);
                            }
                            enc = g.device.create_command_encoder(&Default::default());
                            ctx.slot = 0;
                            let cp2 = enc.begin_compute_pass(&Default::default());
                            cp = cp2;
                        }
                        ctx.bn_silu_f32(&mut cp, &tmp2, &lw.bn_ab, &cb2, te * 512);
                        ctx.gemm_f32(
                            &mut cp, false, &cb2, &lw.pw2, &tmp, tp, 1024, 1024, 1024, 0, 0, 0,
                            1024, 1,
                        );
                    }
                    ctx.add_scaled_f32(&mut cp, &h_cur, &tmp, 1.0, te * 1024);
                }
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("conv") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(enc, &[("h".into(), &h_cur, tp as u64 * 1024)])
                    {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                // ── feed-forward half 2
                ctx.ln_f32(&mut cp, &h_cur, &lw.ln[3], &normed, te);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ln3") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("normed".into(), &normed, tp as u64 * 1024),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.gemm_f32(
                    &mut cp, false, &normed, &lw.ff2a, &ffb, te, 4096, 1024, 4096, 0, 0, 0, 1024, 1,
                );
                ctx.silu_f32(&mut cp, &ffb, te * 4096);
                if l == 0 && std::env::var("BARK3_STAGE").ok().as_deref() == Some("ff2b") {
                    drop(cp);
                    if let Some(o) = self.stage_exit(
                        enc,
                        &[
                            ("ffb".into(), &ffb, tp as u64 * 4096),
                            ("h".into(), &h_cur, tp as u64 * 1024),
                        ],
                    ) {
                        return Ok(o);
                    }
                    enc = g.device.create_command_encoder(&Default::default());
                    ctx.slot = 0;
                    let cp2 = enc.begin_compute_pass(&Default::default());
                    cp = cp2;
                }
                ctx.gemm_f32(
                    &mut cp, false, &ffb, &lw.ff2b, &tmp, te, 1024, 4096, 1024, 0, 0, 0, 4096, 1,
                );
                ctx.add_scaled_f32(&mut cp, &h_cur, &tmp, 0.5, te * 1024);
                // block-final norm_out (inside every conformer block)
                ctx.ln_f32(&mut cp, &h_cur, &lw.ln[4], &h_alt, te);
            }

            layers_in_chunk += 1;
            let need_dump = dump_layer == Some(l);
            if layers_in_chunk >= 3 || need_dump {
                g.queue.submit([enc.finish()]);
                if need_dump {
                    g.wait();
                    let bytes = g.readback(&h_cur, t_real as u64 * 1024 * 4)?;
                    let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                    block_dump = Some(out);
                }
                enc = g.device.create_command_encoder(&Default::default());
                ctx.slot = 0;
                layers_in_chunk = 0;
            }
            std::mem::swap(&mut h_cur, &mut h_alt);
        }

        // epilogue: layer 23's norm_out already applied at its block end; project straight.
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            ctx.gemm_f32(
                &mut cp,
                false,
                &h_cur,
                &Weight::f16(self.w.proj_w.clone()),
                &pooler,
                te,
                640,
                1024,
                640,
                0,
                0,
                0,
                1024,
                1,
            );
            ctx.add_bias_f32(&mut cp, &pooler, &self.w.proj_b, te, 640);
        }
        g.queue.submit([enc.finish()]);
        if let Some(prof) = &self.prof {
            let mut resolve = g.device.create_command_encoder(&Default::default());
            prof.resolve(&mut resolve);
            g.queue.submit([resolve.finish()]);
            g.wait();
            prof.report(g)?;
            prof.reset();
        } else {
            g.wait();
        }
        let out: Vec<f32> = if read_pooler {
            let bytes = g.readback(&pooler, t_real as u64 * 640 * 4)?;
            bytemuck::cast_slice(&bytes).to_vec()
        } else {
            Vec::new()
        };
        Ok(EncodeOut {
            pooler: out,
            pooler_buf: Some(pooler),
            block: block_dump,
        })
    }
}

impl Encoder {
    fn stage_exit(
        &self,
        enc: wgpu::CommandEncoder,
        targets: &[(String, &wgpu::Buffer, u64)],
    ) -> Option<EncodeOut> {
        self.gpu.queue.submit([enc.finish()]);
        pollster::block_on(async { self.gpu.device.poll(wgpu::PollType::wait_indefinitely()) })
            .ok()?;
        let mut block = None;
        for (i, (name, buf, words)) in targets.iter().enumerate() {
            let bytes = self.gpu.readback(buf, words * 4).ok()?;
            let f: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            let nan = f.iter().filter(|v| v.is_nan()).count();
            let w = 1024.min(f.len());
            let rowstats: Vec<(usize, f32)> = (0..6.min(f.len() / w))
                .map(|r| {
                    let row = &f[r * w..(r + 1) * w];
                    let mx = row.iter().fold(0f32, |a, v| a.max(v.abs()));
                    (r, mx)
                })
                .collect();
            eprintln!(
                "[stage-exit {name}] nan={}/{} first={:?} rowmax={:?}",
                nan,
                f.len(),
                &f[..3.min(f.len())],
                rowstats
            );
            if i == 0 {
                block = Some(f);
            }
        }
        Some(EncodeOut {
            pooler: vec![],
            pooler_buf: None,
            block,
        })
    }
}
