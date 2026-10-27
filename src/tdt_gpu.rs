//! GPU-resident TDT greedy decoder.
//!
//! The joint head, both LSTM layers, and the projection are each split across
//! many workgroups. Layer 0 also reduces the joint partials and owns the frame
//! control flow. Decoder state, frame pointer, and emitted tokens never leave
//! VRAM. Dispatch chunks are submitted together and only the small result
//! buffer is read back.

use anyhow::{Result, bail, ensure};

use crate::cfg;
use crate::gpu::Gpu;
use crate::tdt::{TdtDecoder, Token};

const SHADER: &str = include_str!("tdt_gpu/shader.wgsl");

const DH: u32 = cfg::DEC_HIDDEN as u32;
const VOCAB: u32 = cfg::VOCAB as u32;
const N_ROWS: u32 = opts_vocab_rows();
const JOINT_GROUPS: u32 = 65;
/// Must match `LAYER_GROUPS` in `tdt_gpu/shader.wgsl`.
const LAYER_GROUPS: u32 = 40;
const CHANNELS_PER_GROUP: u32 = DH / LAYER_GROUPS;
// The shader assumes every workgroup owns a whole number of 16-lane row groups
// and that the weight arena is addressable as `vec4<u32>`.
const _: () = assert!(DH % LAYER_GROUPS == 0);
const _: () = assert!((4 * CHANNELS_PER_GROUP) % 16 == 0);
const _: () = assert!(LANES == 16);
const CHUNK_STEPS: u32 = 256;
/// u32 words per token record in the result buffer: id, frame, duration.
/// Must match `RESULT_TOKEN_BASE + count * 3u` in `tdt_gpu/shader.wgsl`.
const TOKEN_WORDS: u32 = 3;

const W_HH: u32 = 1_638_400;
const W_HEAD: u32 = 3_276_800;
const W_PROJ: u32 = 5_900_160;
const B_HH: u32 = 5_120;
const B_HEAD: u32 = 10_240;
const B_PROJ: u32 = 18_438;
const STATE_LEN: u32 = 5_120;
const S_H0A: usize = 0;
const S_H1A: usize = 1_280;
const S_C0: usize = 2_560;
const S_C1: usize = 3_200;
const S_DEC: usize = 3_840;

const fn opts_vocab_rows() -> u32 {
    cfg::VOCAB as u32 + cfg::N_DUR as u32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    t_len: u32,
    max_steps: u32,
    _a: u32,
    _b: u32,
}

/// Stats from one GPU decode, useful for diagnostics and stage accounting.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuDecodeStats {
    pub steps: u32,
    pub final_frame: u32,
    pub chunks: u32,
    pub exit_code: u32,
    pub record_ms: f32,
    pub wait_ms: f32,
    pub total_ms: f32,
}

pub struct GpuDecodeOutput {
    pub tokens: Vec<Token>,
    pub stats: GpuDecodeStats,
}

pub struct GpuTdtDecoder {
    embedding: wgpu::Buffer,
    weights: wgpu::Buffer,
    biases: wgpu::Buffer,
    initial_state: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    init_pipeline: wgpu::ComputePipeline,
    joint_pipeline: wgpu::ComputePipeline,
    layer0_pipeline: wgpu::ComputePipeline,
    layer1_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
}

impl GpuTdtDecoder {
    pub fn new(gpu: &Gpu, tdt: &TdtDecoder) -> Result<Self> {
        let mut weights = Vec::with_capacity(6_104_960);
        pack_f16_lanes(&tdt.w_ih, &mut weights)?;
        ensure!(weights.len() as u32 == W_HH, "unexpected w_ih size");
        pack_f16_lanes(&tdt.w_hh, &mut weights)?;
        ensure!(weights.len() as u32 == W_HEAD, "unexpected w_hh size");
        pack_f16_lanes(&tdt.head_w, &mut weights)?;
        ensure!(weights.len() as u32 == W_PROJ, "unexpected head size");
        pack_f16_lanes(&tdt.proj_w, &mut weights)?;
        ensure!(
            weights.len() as u32 == 6_104_960,
            "unexpected decoder weight size"
        );
        ensure!(
            weights.len() % 4 == 0,
            "decoder weight arena is not vec4 aligned"
        );

        let mut biases = Vec::with_capacity(19_078);
        biases.extend_from_slice(&tdt.b_ih);
        ensure!(biases.len() as u32 == B_HH, "unexpected b_ih size");
        biases.extend_from_slice(&tdt.b_hh);
        ensure!(biases.len() as u32 == B_HEAD, "unexpected b_hh size");
        biases.extend_from_slice(&tdt.head_b);
        ensure!(biases.len() as u32 == B_PROJ, "unexpected head bias size");
        biases.extend_from_slice(&tdt.proj_b);
        ensure!(
            biases.len() as u32 == 19_078,
            "unexpected decoder bias size"
        );
        ensure!(
            tdt.embed.len() == VOCAB as usize * DH as usize,
            "unexpected decoder embedding size"
        );
        ensure!(
            JOINT_GROUPS == N_ROWS.div_ceil(128),
            "joint workgroup count does not match the shader"
        );

        // The blank-start decoder state is audio-independent. Compute it once
        // with the CPU reference and seed every GPU decode from this template.
        let (initial_h, initial_c, initial_dec) = tdt.init_state();
        ensure!(
            initial_h.len() == 2 * DH as usize
                && initial_c.len() == 2 * DH as usize
                && initial_dec.len() == DH as usize,
            "unexpected initial decoder state size"
        );
        let mut initial_state = vec![0f32; STATE_LEN as usize];
        initial_state[S_H0A..S_H0A + DH as usize].copy_from_slice(&initial_h[..DH as usize]);
        initial_state[S_H1A..S_H1A + DH as usize].copy_from_slice(&initial_h[DH as usize..]);
        initial_state[S_C0..S_C0 + DH as usize].copy_from_slice(&initial_c[..DH as usize]);
        initial_state[S_C1..S_C1 + DH as usize].copy_from_slice(&initial_c[DH as usize..]);
        initial_state[S_DEC..S_DEC + DH as usize].copy_from_slice(&initial_dec);

        let mut uploader = gpu.uploader();
        let weight_buf = gpu.storage("tdt_weights", weights.len() as u64 * 4);
        uploader.upload(&weight_buf, bytemuck::cast_slice(&weights))?;
        let bias_buf = gpu.storage("tdt_biases", biases.len() as u64 * 4);
        uploader.upload(&bias_buf, bytemuck::cast_slice(&biases))?;
        let embed_buf = gpu.storage("tdt_embedding", tdt.embed.len() as u64 * 4);
        uploader.upload(&embed_buf, bytemuck::cast_slice(&tdt.embed))?;
        let initial_buf = gpu.storage("tdt_initial_state", STATE_LEN as u64 * 4);
        uploader.upload(&initial_buf, bytemuck::cast_slice(&initial_state))?;
        uploader.finish()?;

        let bind_group_layout = create_bind_group_layout(gpu);
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tdt_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                ..Default::default()
            });
        let init_pipeline =
            gpu.pipeline("tdt_init", SHADER, "init_main", Some(&pipeline_layout))?;
        let joint_pipeline =
            gpu.pipeline("tdt_joint", SHADER, "joint_main", Some(&pipeline_layout))?;
        let layer0_pipeline =
            gpu.pipeline("tdt_layer0", SHADER, "layer0_main", Some(&pipeline_layout))?;
        let layer1_pipeline =
            gpu.pipeline("tdt_layer1", SHADER, "layer1_main", Some(&pipeline_layout))?;
        let project_pipeline = gpu.pipeline(
            "tdt_project",
            SHADER,
            "project_main",
            Some(&pipeline_layout),
        )?;
        Ok(Self {
            embedding: embed_buf,
            weights: weight_buf,
            biases: bias_buf,
            initial_state: initial_buf,
            bind_group_layout,
            init_pipeline,
            joint_pipeline,
            layer0_pipeline,
            layer1_pipeline,
            project_pipeline,
        })
    }

    pub fn decode(
        &self,
        gpu: &Gpu,
        pooler: &wgpu::Buffer,
        t_len: usize,
    ) -> Result<GpuDecodeOutput> {
        if t_len == 0 {
            return Ok(GpuDecodeOutput {
                tokens: Vec::new(),
                stats: GpuDecodeStats::default(),
            });
        }
        let decode_started = std::time::Instant::now();
        let max_steps = t_len
            .saturating_mul(cfg::MAX_SYMBOLS as usize + 4)
            .saturating_add(64);
        ensure!(
            max_steps <= u32::MAX as usize,
            "audio is too long for the GPU decoder"
        );
        let max_steps = max_steps as u32;
        let result_words = 6u32 + TOKEN_WORDS * max_steps;
        let result_bytes = result_words as u64 * 4;

        let state = gpu.storage("tdt_state", STATE_LEN as u64 * 4);
        let result = gpu.storage("tdt_result", result_bytes);
        let partials = gpu.storage("tdt_partials", JOINT_GROUPS as u64 * 16);
        let params = gpu.uniform("tdt_params", std::mem::size_of::<Params>() as u64);
        gpu.write_at(&result, 0, &[0u8; 24]);
        gpu.write_at(
            &params,
            0,
            bytemuck::bytes_of(&Params {
                t_len: t_len as u32,
                max_steps,
                _a: 0,
                _b: 0,
            }),
        );

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tdt_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pooler.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.embedding.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.biases.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let profile_enabled = std::env::var("BARK3_GPUDEC_PROFILE").is_ok()
            && gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && gpu
                .features
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let profile_steps = 32u32;
        let profile_query_count = 1 + profile_steps * 8;
        let profile = profile_enabled.then(|| {
            (
                gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("tdt_profile_queries"),
                    ty: wgpu::QueryType::Timestamp,
                    count: profile_query_count,
                }),
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tdt_profile_ticks"),
                    size: profile_query_count as u64 * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            )
        });
        let mut profile_totals = [0f64; 4];

        let mut steps_done = 0u32;
        let mut chunks = 0u32;
        let mut last_bytes = Vec::new();
        let mut record_seconds = 0f32;
        let mut wait_seconds = 0f32;
        loop {
            let remaining = max_steps.saturating_sub(steps_done);
            if remaining == 0 {
                break;
            }
            let steps = remaining.min(CHUNK_STEPS);
            let is_first_chunk = chunks == 0;
            let profiled_steps = if profile.is_some() && is_first_chunk {
                steps.min(profile_steps)
            } else {
                0
            };
            let record_started = std::time::Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            if is_first_chunk {
                encoder.copy_buffer_to_buffer(
                    &self.initial_state,
                    0,
                    &state,
                    0,
                    STATE_LEN as u64 * 4,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &bind_group, &[]);
                if let Some((queries, _)) = &profile {
                    if is_first_chunk {
                        pass.write_timestamp(queries, 0);
                    }
                }
                if is_first_chunk {
                    pass.set_pipeline(&self.init_pipeline);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                for step_index in 0..steps {
                    if step_index < profiled_steps {
                        let queries = &profile.as_ref().unwrap().0;
                        let base = 1 + step_index * 8;
                        pass.write_timestamp(queries, base);
                        pass.set_pipeline(&self.joint_pipeline);
                        pass.dispatch_workgroups(JOINT_GROUPS, 1, 1);
                        pass.write_timestamp(queries, base + 1);
                        pass.set_pipeline(&self.layer0_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                        pass.write_timestamp(queries, base + 2);
                        pass.set_pipeline(&self.layer1_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                        pass.write_timestamp(queries, base + 3);
                        pass.set_pipeline(&self.project_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                        pass.write_timestamp(queries, base + 4);
                    } else {
                        pass.set_pipeline(&self.joint_pipeline);
                        pass.dispatch_workgroups(JOINT_GROUPS, 1, 1);
                        pass.set_pipeline(&self.layer0_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                        pass.set_pipeline(&self.layer1_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                        pass.set_pipeline(&self.project_pipeline);
                        pass.dispatch_workgroups(LAYER_GROUPS, 1, 1);
                    }
                }
            }
            if profiled_steps > 0 {
                let (queries, ticks) = profile.as_ref().unwrap();
                encoder.resolve_query_set(queries, 0..(1 + profiled_steps * 8), ticks, 0);
            }
            gpu.queue.submit([encoder.finish()]);
            if profiled_steps > 0 {
                let (_, ticks) = profile.as_ref().unwrap();
                let bytes = gpu.readback(ticks, (1 + profiled_steps * 8) as u64 * 8)?;
                let period = gpu.queue.get_timestamp_period() as f64;
                for step_index in 0..profiled_steps as usize {
                    let base = 1 + step_index * 8;
                    for (stage, total) in profile_totals.iter_mut().enumerate() {
                        // The writer emits five consecutive timestamps per step
                        // (one per stage boundary), so stage i owns the span
                        // [base + i, base + i + 1).
                        let start = read_u64(&bytes, (base + stage) * 8) as f64;
                        let end = read_u64(&bytes, (base + stage + 1) * 8) as f64;
                        *total += (end - start) * period;
                    }
                }
            }
            record_seconds += record_started.elapsed().as_secs_f32();
            chunks += 1;

            let wait_started = std::time::Instant::now();
            let bytes = gpu.readback(&result, result_bytes)?;
            wait_seconds += wait_started.elapsed().as_secs_f32();
            let header = result_header(&bytes);
            last_bytes = bytes;
            if header[3] != 0 || header[2] >= max_steps {
                break;
            }
            if header[2] <= steps_done {
                bail!(
                    "GPU TDT decoder made no progress: steps={} frame={}",
                    header[2],
                    header[1]
                );
            }
            steps_done = header[2];
        }

        let header = result_header(&last_bytes);
        let token_count = (header[0] as usize).min(max_steps as usize);
        let mut tokens = Vec::with_capacity(token_count);
        for index in 0..token_count {
            let base = (6 + TOKEN_WORDS as usize * index) * 4;
            tokens.push(Token {
                id: read_u32(&last_bytes, base),
                frame: read_u32(&last_bytes, base + 4),
                dur: read_u32(&last_bytes, base + 8),
            });
        }
        if profile_enabled && profile_steps > 0 {
            let n = profile_steps.min(header[2].max(1)) as f64;
            println!(
                "[gpu-profile] per step: joint={:.1}us layer0={:.1}us layer1={:.1}us project={:.1}us (n={n})",
                profile_totals[0] / n / 1000.0,
                profile_totals[1] / n / 1000.0,
                profile_totals[2] / n / 1000.0,
                profile_totals[3] / n / 1000.0,
            );
        }
        let stats = GpuDecodeStats {
            steps: header[2],
            final_frame: header[1],
            chunks,
            exit_code: header[3],
            record_ms: record_seconds * 1000.0,
            wait_ms: wait_seconds * 1000.0,
            total_ms: decode_started.elapsed().as_secs_f32() * 1000.0,
        };
        Ok(GpuDecodeOutput { tokens, stats })
    }
}

/// Lanes per row and the K values each lane owns.
const LANES: usize = 16;
const LANE_SPAN: usize = DH as usize / LANES;

/// Repack a row-major f16 matrix so each 16-lane group finds its K values
/// contiguously. Lane `j` owns K residues `j, j + 16, j + 32, ...` and walks them
/// in that order, so gathering them into one 40-element block per lane keeps the
/// per-lane accumulation sequence bit-identical while turning 40 scattered
/// 4-byte loads into five `vec4` fetches.
fn pack_f16_lanes(source: &[u16], out: &mut Vec<u32>) -> Result<()> {
    let width = DH as usize;
    ensure!(source.len() % width == 0, "f16 tensor is not row aligned");
    ensure!(
        LANE_SPAN == 40,
        "lane span must stay in sync with the shader"
    );
    let start = out.len();
    out.resize(start + source.len() / 2, 0);
    let words = &mut out[start..];
    let lane_words = LANE_SPAN / 2;
    for (r, row) in source.chunks_exact(width).enumerate() {
        let base = r * (width / 2);
        for j in 0..LANES {
            let lane_base = base + j * lane_words;
            for m in 0..LANE_SPAN {
                let value = row[j + LANES * m] as u32;
                if m % 2 == 0 {
                    words[lane_base + m / 2] = value;
                } else {
                    words[lane_base + m / 2] |= value << 16;
                }
            }
        }
    }
    Ok(())
}

fn create_bind_group_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    gpu.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_bind_group_layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
}

fn result_header(bytes: &[u8]) -> [u32; 4] {
    [
        read_u32(bytes, 0),
        read_u32(bytes, 4),
        read_u32(bytes, 8),
        read_u32(bytes, 12),
    ]
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
