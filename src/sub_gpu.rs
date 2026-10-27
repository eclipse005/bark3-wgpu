//! GPU conv subsampling: mel -> conv3x3(stride2, ReLU) -> dw3x3(stride2)*2
//! -> pointwise(ReLU)*2 -> flatten (T, 4096).  Everything stays in VRAM; the
//! encoder consumes the resulting buffer directly (no host round-trip).
//!
//! Layouts are channel-major (c, t, m) throughout, like the CPU reference.

// Dispatch helpers intentionally expose every shader dimension and buffer
// explicitly; grouping them into structs would hide the launch contract.
#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;
use crate::sub::SubsamplingWeights;

fn conv_len(l: usize) -> usize {
    (l + 2 - 3) / 2 + 1
}

const IM2COL: &str = r#"
struct Cfg { f: u32, f1: u32, m1: u32, _a: u32, p: u32, _b: u32, _c: u32, _d: u32 };
@group(0) @binding(0) var<storage, read>       mel: array<f32>;
@group(0) @binding(1) var<storage, read_write> A:   array<f32>;   // (16, p)
@group(0) @binding(2) var<uniform>             cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 16u * cfg.p) { return; }
    let t = idx / cfg.p;
    let p = idx % cfg.p;
    var v = 0.0;
    if (t < 9u) {
        let i = p / cfg.m1;
        let j = p % cfg.m1;
        let ii = 2u * i + t / 3u;
        let jj = 2u * j + t % 3u;
        if (ii >= 1u && ii <= 2u * cfg.f && jj >= 1u && jj <= 256u) {
            v = mel[(ii - 1u) * 128u + (jj - 1u)];
        }
    }
    A[idx] = v;
}
"#;

const DWCONV: &str = r#"
struct Cfg { t_in: u32, m_in: u32, t_out: u32, m_out: u32, vlen: u32, stride: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       w:   array<f32>;   // (256, 9)
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wid.y;
    let p = wid.x * 256u + lid.x;
    let np = cfg.t_out * cfg.m_out;
    if (p >= np) { return; }
    let i = p / cfg.m_out;
    let j = p % cfg.m_out;
    let base = c * cfg.t_in * cfg.m_in;
    var s = b[c];
    for (var ki: i32 = 0; ki < 3; ki = ki + 1) {
        let ii = i32(i) * 2 + ki - 1;
        if (ii < 0 || ii >= i32(cfg.t_in)) { continue; }
        for (var kj: i32 = 0; kj < 3; kj = kj + 1) {
            let jj = i32(j) * 2 + kj - 1;
            if (jj < 0 || jj >= i32(cfg.m_in)) { continue; }
            s = s + w[c * 9u + u32(ki * 3 + kj)] * x[base + u32(ii) * cfg.m_in + u32(jj)];
        }
    }
    var o = s;
    if (i >= cfg.vlen) { o = 0.0; }
    out[c * cfg.stride + p] = o;
}
"#;

// C[n, m] = A^T[m, k] * W^T[k, n] + bias[n], A column-major (k, m),
// W row-major f16 (k, n), C column-major (n, m).  64x64 tile, 8x8 per thread.
// Accumulators are emitted as 64 scalars: a nested array would be indexed
// dynamically and spilled to local memory by the WGSL compiler.
fn pw_shader() -> String {
    let mut s = String::new();
    s.push_str(
        "struct Cfg { m: u32, n: u32, k: u32, vlen: u32, rdiv: u32, _a: u32, _b: u32, _c: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A:    array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W:    array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C:    array<f32>;\n\
         @group(0) @binding(3) var<storage, read>       bias: array<f32>;\n\
         @group(0) @binding(4) var<uniform>             cfg:  Cfg;\n\
         var<workgroup> As: array<vec4<f32>, 256>;\n\
         var<workgroup> Bs: array<vec4<u32>, 128>;\n\
         @compute @workgroup_size(8, 8)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.x * 64u;\n let n0 = wid.y * 64u;\n\
         let tid = ty * 8u + tx;\n",
    );
    for i in 0..8 {
        for j in 0..8 {
            s.push_str(&format!("var c{i}{j}: f32 = 0.0;\n"));
        }
    }
    s.push_str(
        "var k0: u32 = 0u;\n loop {\n  if (k0 >= cfg.k) { break; }\n\
         for (var t: u32 = tid; t < 256u; t = t + 64u) {\n\
           let kk = t / 16u;\n           let q = t % 16u;\n           let p = m0 + q * 4u;\n\
           // m is not always a multiple of 64: the first pointwise layer of the\n\
           // sub-sampler has m = t1 * m2 = 131.5 * 64. That tail quad used to\n\
           // read past its channel -- and, for the last channel, past the whole\n\
           // buffer, which made the front end non-deterministic -- and its rows\n\
           // were then stored over the next channel. Read zeros instead; the\n\
           // store below drops the rows that do not exist.\n\
           if (p + 4u <= cfg.m) {\n\
             As[t] = vec4<f32>(A[(k0 + kk) * cfg.m + p], A[(k0 + kk) * cfg.m + p + 1u],\n\
                               A[(k0 + kk) * cfg.m + p + 2u], A[(k0 + kk) * cfg.m + p + 3u]);\n\
           } else {\n\
             As[t] = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n\
           }\n\
         }\n\
         for (var t: u32 = tid; t < 128u; t = t + 64u) {\n\
           let kk = t / 8u;\n           let q = t % 8u;\n\
           let base = (k0 + kk) * (cfg.n / 2u) + n0 / 2u + q * 4u;\n\
           Bs[t] = vec4<u32>(W[base], W[base + 1u], W[base + 2u], W[base + 3u]);\n\
         }\n\
         workgroupBarrier();\n\
         for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {\n\
           let a0 = As[kk * 16u + ty * 2u];\n           let a1 = As[kk * 16u + ty * 2u + 1u];\n\
           let bw = Bs[kk * 8u + tx];\n\
           let b0 = unpack2x16float(bw.x);\n           let b1 = unpack2x16float(bw.y);\n\
           let b2 = unpack2x16float(bw.z);\n           let b3 = unpack2x16float(bw.w);\n",
    );
    for i in 0..8 {
        s.push_str(&format!(
            "let av{i} = select(a0[{c}], a1[{c}], {});\n",
            if i >= 4 { "true" } else { "false" },
            c = i % 4
        ));
    }
    for i in 0..8 {
        for j in 0..8 {
            let b = match j {
                0 => "b0.x".to_string(),
                1 => "b0.y".to_string(),
                2 => "b1.x".to_string(),
                3 => "b1.y".to_string(),
                4 => "b2.x".to_string(),
                5 => "b2.y".to_string(),
                6 => "b3.x".to_string(),
                _ => "b3.y".to_string(),
            };
            s.push_str(&format!("c{i}{j} = c{i}{j} + av{i} * {b};\n"));
        }
    }
    s.push_str("         }\n         workgroupBarrier();\n         k0 = k0 + 16u;\n }\n");
    for i in 0..8 {
        s.push_str(&format!("let row{i} = m0 + ty * 8u + {i}u;\n"));
    }
    for j in 0..8 {
        s.push_str(&format!("let col{j} = n0 + tx * 8u + {j}u;\n"));
    }
    for i in 0..8 {
        for j in 0..8 {
            s.push_str(&format!(
                "var o{i}{j} = max(c{i}{j} + bias[col{j}], 0.0);\nif (row{i} / cfg.rdiv >= cfg.vlen) {{ o{i}{j} = 0.0; }}\nif (row{i} < cfg.m) {{ C[col{j} * cfg.m + row{i}] = o{i}{j}; }}\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

const FLATTEN: &str = r#"
struct Cfg { t_real: u32, tp: u32, mlen: u32, cm: u32, stride: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       x:   array<f32>;   // (cm, stride)
@group(0) @binding(1) var<storage, read_write> out: array<f32>;   // (tp, cm*mlen)
@group(0) @binding(2) var<uniform>             cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let width = cfg.cm * cfg.mlen;
    if (idx >= cfg.tp * width) { return; }
    let t = idx / width;
    let r = idx % width;
    let c = r / cfg.mlen;
    let m = r % cfg.mlen;
    var v = 0.0;
    if (t < cfg.t_real) {
        v = x[c * cfg.stride + t * cfg.mlen + m];
    }
    out[idx] = v;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cfg8 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    f: u32,
    g: u32,
    h: u32,
}

pub struct SubGpu {
    dw: wgpu::ComputePipeline,
    pw: wgpu::ComputePipeline,
    flat: wgpu::ComputePipeline,
    bgl4: wgpu::BindGroupLayout,
    bgl_pw: wgpu::BindGroupLayout,
    bgl_flat: wgpu::BindGroupLayout,
    w_l0t: wgpu::Buffer, // (k=16, n=256) f16, im2col conv as a GEMM
    b_l0: wgpu::Buffer,
    im2col: wgpu::ComputePipeline,
    bgl_im2col: wgpu::BindGroupLayout,
    w_l2: wgpu::Buffer,
    b_l2: wgpu::Buffer,
    w_l3t: wgpu::Buffer,
    b_l3: wgpu::Buffer,
    w_l5: wgpu::Buffer,
    b_l5: wgpu::Buffer,
    w_l6t: wgpu::Buffer,
    b_l6: wgpu::Buffer,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uni_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SubGpu {
    pub fn new(g: &Gpu, sw: &SubsamplingWeights) -> Result<Self> {
        let pipe = |label: &str,
                    src: &str,
                    entries: &[wgpu::BindGroupLayoutEntry]|
         -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
            let bgl = g
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries,
                });
            let pl = g
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    ..Default::default()
                });
            let p = g.pipeline(label, src, "main", Some(&pl))?;
            Ok((p, bgl))
        };
        let bgl4 = g
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sub_4in"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    uni_entry(4),
                ],
            });
        let (dw, _) = pipe(
            "sub_dw",
            DWCONV,
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uni_entry(4),
            ],
        )?;
        let (pw, bgl_pw) = pipe(
            "sub_pw",
            &pw_shader(),
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                uni_entry(4),
            ],
        )?;
        let (flat, bgl_flat) = pipe(
            "sub_flat",
            FLATTEN,
            &[
                storage_entry(0, true),
                storage_entry(1, false),
                uni_entry(2),
            ],
        )?;
        let (im2col, bgl_im2col) = pipe(
            "sub_im2col",
            IM2COL,
            &[
                storage_entry(0, true),
                storage_entry(1, false),
                uni_entry(2),
            ],
        )?;
        let up = |label: &str, data: &[f32]| -> wgpu::Buffer {
            let b = g.storage(label, (data.len() * 4) as u64);
            g.upload(&b, bytemuck::cast_slice(data));
            b
        };
        // pointwise weights consumed as W^T[k, n] f16
        let wt = |label: &str, w: &[f32]| -> wgpu::Buffer {
            let (n, k) = (256usize, 256usize);
            let mut t = vec![0f32; k * n];
            for co in 0..n {
                for ci in 0..k {
                    t[ci * n + co] = w[co * k + ci];
                }
            }
            let b = g.storage(label, (t.len() * 2) as u64);
            g.upload(
                &b,
                bytemuck::cast_slice(&crate::raw::RawTensor::f32_to_f16_words(&t)),
            );
            b
        };
        // conv1 as GEMM: W^T[t=16, c=256] f16, zero-padded taps 9..15
        let mut w0t = vec![0f32; 16 * 256];
        for c in 0..256 {
            for t in 0..9 {
                w0t[t * 256 + c] = sw.l0_w[c * 9 + t];
            }
        }
        let w0b = g.storage("sub_wl0t", (w0t.len() * 2) as u64);
        g.upload(
            &w0b,
            bytemuck::cast_slice(&crate::raw::RawTensor::f32_to_f16_words(&w0t)),
        );
        Ok(Self {
            dw,
            pw,
            flat,
            bgl4,
            bgl_pw,
            bgl_flat,
            im2col,
            bgl_im2col,
            w_l0t: w0b,
            b_l0: up("sub_bl0", &sw.l0_b),
            w_l2: up("sub_wl2", &sw.l2_w),
            b_l2: up("sub_bl2", &sw.l2_b),
            w_l3t: wt("sub_wl3", &sw.l3_w),
            b_l3: up("sub_bl3", &sw.l3_b),
            w_l5: up("sub_wl5", &sw.l5_w),
            b_l5: up("sub_bl5", &sw.l5_b),
            w_l6t: wt("sub_wl6", &sw.l6_w),
            b_l6: up("sub_bl6", &sw.l6_b),
        })
    }

    fn uniform(&self, g: &Gpu, c: Cfg8) -> wgpu::Buffer {
        let b = g.uniform("sub_cfg", 32);
        g.upload(&b, bytemuck::bytes_of(&c));
        b
    }

    /// `BARK3_DUMP_SUBX=<name>`: write a stage buffer to `data/dump_sub_<name>.bin`.
    fn dump_stage(g: &Gpu, name: &str, buf: &wgpu::Buffer, bytes: u64) {
        if std::env::var("BARK3_DUMP_SUBX").is_err() {
            return;
        }
        if let Ok(b) = g.readback(buf, bytes) {
            std::fs::write(format!("data/dump_sub_{name}.bin"), &b).ok();
            eprintln!("[subx] dumped {name} ({} bytes)", b.len());
        }
    }

    fn dw_dispatch(
        &self,
        g: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        t_in: usize,
        m_in: usize,
        t_out: usize,
        m_out: usize,
        vlen: usize,
        out_stride: usize,
    ) {
        let cfg = self.uniform(
            g,
            Cfg8 {
                a: t_in as u32,
                b: m_in as u32,
                c: t_out as u32,
                d: m_out as u32,
                e: vlen as u32,
                f: out_stride as u32,
                g: 0,
                h: 0,
            },
        );
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl4,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg.as_entire_binding(),
                },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.dw);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(((t_out * m_out).div_ceil(256)) as u32, 256, 1);
    }

    fn im2col_dispatch(
        &self,
        g: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        mel: &wgpu::Buffer,
        a: &wgpu::Buffer,
        f: usize,
        f1: usize,
        m1: usize,
        p: usize,
    ) {
        let cfg = self.uniform(
            g,
            Cfg8 {
                a: f as u32,
                b: f1 as u32,
                c: m1 as u32,
                d: 0,
                e: p as u32,
                f: 0,
                g: 0,
                h: 0,
            },
        );
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_im2col,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg.as_entire_binding(),
                },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.im2col);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(((16 * p).div_ceil(256)) as u32, 1, 1);
    }

    fn pw_dispatch(
        &self,
        g: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        wt: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        out: &wgpu::Buffer,
        m: usize,
        k: usize,
        vlen: u32,
        rdiv: u32,
    ) {
        let cfg = self.uniform(
            g,
            Cfg8 {
                a: m as u32,
                b: 256,
                c: k as u32,
                d: vlen,
                e: rdiv,
                f: 0,
                g: 0,
                h: 0,
            },
        );
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_pw,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg.as_entire_binding(),
                },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.pw);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((m.div_ceil(64)) as u32, 4, 1);
    }

    /// Run the whole front end.  Returns a (pad128(T), 4096) f32 buffer ready for
    /// the encoder's subsampling projection.
    pub fn run(
        &self,
        g: &Gpu,
        mel: &[f32],
        n_frames: usize,
        n_valid: usize,
    ) -> Result<(wgpu::Buffer, usize)> {
        let f = n_frames;
        let m0 = 128usize;
        let f1 = conv_len(f);
        let m1 = conv_len(m0);
        let t1 = conv_len(f1);
        let m2 = conv_len(m1);
        let t2 = conv_len(t1);
        let m3 = conv_len(m2);
        let v0 = conv_len(n_valid.max(1));
        let v1 = conv_len(v0);
        let v2 = conv_len(v1);
        let t_real = t2;

        let prof = std::env::var("BARK3_SUBG_TIME").is_ok();
        let ta = crate::clock::now();
        let mel_b = g.storage("sub_mel", (mel.len() * 4) as u64);
        g.upload(&mel_b, bytemuck::cast_slice(mel));

        let x1 = g.storage("sub_x1", (256 * f1 * m1 * 4) as u64);
        let p1 = t1 * m2;
        let x2 = g.storage("sub_x2", (256 * p1 * 4) as u64);
        let x3 = g.storage("sub_x3", (256 * p1 * 4) as u64);
        let p2 = (t2 * m3).div_ceil(64) * 64;
        let x4 = g.storage("sub_x4", (256 * p2 * 4) as u64);
        let x5 = g.storage("sub_x5", (256 * p2 * 4) as u64);
        let tp = crate::encoder::pad128(t_real);
        let out = g.storage("sub_out", (tp * 4096 * 4) as u64);
        let t_alloc = ta.elapsed();
        if prof {
            let e0 = g.device.create_command_encoder(&Default::default());
            g.queue.submit([e0.finish()]);
            g.wait();
            eprintln!(
                "[subg] empty-submit after alloc {:.3}s",
                ta.elapsed().as_secs_f32()
            );
        }
        let ta2 = crate::clock::now();
        let p_conv = f1 * m1;
        let patch = g.storage("sub_patch", (16 * p_conv * 4) as u64);
        let mut enc = g.device.create_command_encoder(&Default::default());
        let t0 = crate::clock::now();
        self.im2col_dispatch(g, &mut enc, &mel_b, &patch, f, f1, m1, p_conv);
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] im2col#1 {:.3}s", t0.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
            let t9 = crate::clock::now();
            self.im2col_dispatch(g, &mut enc, &mel_b, &patch, f, f1, m1, p_conv);
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] im2col#2 {:.3}s", t9.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        let tc = crate::clock::now();
        self.pw_dispatch(
            g,
            &mut enc,
            &patch,
            &self.w_l0t,
            &self.b_l0,
            &x1,
            p_conv,
            16,
            v0 as u32,
            m1 as u32,
        );
        Self::dump_stage(g, "x1", &x1, (256 * p_conv * 4) as u64);
        let t_conv1 = tc.elapsed();
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] conv1_pw {:.3}s", t_conv1.as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        Self::dump_stage(g, "x1b", &x1, (256 * p_conv * 4) as u64);
        let t1s = crate::clock::now();
        self.dw_dispatch(
            g, &mut enc, &x1, &x2, &self.w_l2, &self.b_l2, f1, m1, t1, m2, v1, p1,
        );
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] dw1 {:.3}s", t1s.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        let t2s = crate::clock::now();
        self.pw_dispatch(
            g,
            &mut enc,
            &x2,
            &self.w_l3t,
            &self.b_l3,
            &x3,
            p1,
            256,
            u32::MAX,
            1,
        );
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] pw1 {:.3}s", t2s.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        let t3s = crate::clock::now();
        self.dw_dispatch(
            g, &mut enc, &x3, &x4, &self.w_l5, &self.b_l5, t1, m2, t2, m3, v2, p2,
        );
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] dw2 {:.3}s", t3s.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        let t4s = crate::clock::now();
        self.pw_dispatch(
            g,
            &mut enc,
            &x4,
            &self.w_l6t,
            &self.b_l6,
            &x5,
            p2,
            256,
            u32::MAX,
            1,
        );
        if prof {
            g.queue.submit([enc.finish()]);
            g.wait();
            eprintln!("[subg] pw2 {:.3}s", t4s.elapsed().as_secs_f32());
            enc = g.device.create_command_encoder(&Default::default());
        }
        {
            let cfg = self.uniform(
                g,
                Cfg8 {
                    a: t_real as u32,
                    b: tp as u32,
                    c: m3 as u32,
                    d: 256,
                    e: p2 as u32,
                    f: 0,
                    g: 0,
                    h: 0,
                },
            );
            let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl_flat,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x5.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cfg.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.flat);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(((tp * 4096).div_ceil(256)) as u32, 1, 1);
        }
        g.queue.submit([enc.finish()]);
        if std::env::var("BARK3_DUMP_SUBX").is_ok() {
            g.flush()?;
            Self::dump_stage(g, "x1", &x1, (256 * f1 * m1 * 4) as u64);
            Self::dump_stage(g, "x2", &x2, (256 * p1 * 4) as u64);
            Self::dump_stage(g, "x3", &x3, (256 * p1 * 4) as u64);
            Self::dump_stage(g, "x4", &x4, (256 * p2 * 4) as u64);
            Self::dump_stage(g, "x5", &x5, (256 * p2 * 4) as u64);
            Self::dump_stage(g, "out", &out, (tp * 4096 * 4) as u64);
        }
        if prof {
            g.wait();
            eprintln!(
                "[subg] kernels {:.3}s alloc {:.3}s record {:.3}s (conv1 build {:.3}s) x1={}MB",
                t0.elapsed().as_secs_f32(),
                t_alloc.as_secs_f32(),
                ta2.elapsed().as_secs_f32(),
                t_conv1.as_secs_f32(),
                x1.size() / 1024 / 1024
            );
        }
        Ok((out, t_real))
    }
}
