use anyhow::{Context, Result};

use crate::encoder::EncodeOut;
use crate::inference::{AsrInference, TranscribeOptions};
use crate::sub::subsample;
use crate::transcript::{TranscribeResult, vocab};

/// Run the production transcription pipeline without any diagnostic command logic.
pub(crate) fn run(
    asr: &AsrInference,
    wave: &[f32],
    dump_layer: Option<usize>,
    options: &TranscribeOptions,
) -> Result<TranscribeResult> {
    let AsrInference {
        gpu,
        tdt,
        sub_w,
        cpu,
        mel_fe,
        ..
    } = asr;
    // A CPU-only process has no adapter at all, so the backend choice cannot
    // drift from the environment: no GPU state means the CPU encoder must run.
    let cpu_path = gpu.is_none();
    // Default: the decoder is GPU-resident as well, so the wgpu path never runs
    // TDT arithmetic on the CPU. BARK3_CPU_DEC=1 selects the CPU reference
    // decoder for comparison.
    let use_gpu_decoder = !cpu_path && std::env::var("BARK3_CPU_DEC").is_err();
    let dump_pool = std::env::var("BARK3_DUMP_POOL").is_ok();

    // Timestamps belong to the file the caller handed us, so remember its
    // length before the reference pipeline prepends its 0.25 s of silence.
    let audio_seconds = wave.len() as f32 / 16_000.0;

    // Official inference pads 0.25 s of silence.
    let mut wave = wave.to_vec();
    let mut padded = vec![0f32; 4000];
    padded.append(&mut wave);
    let wave = &padded;

    let t0 = crate::clock::now();
    let feats = mel_fe.compute(wave);
    let t_mel = t0.elapsed().as_secs_f32();

    let t1 = crate::clock::now();
    let dump_layer_env: Option<usize> = std::env::var("BARK3_DUMP_L")
        .ok()
        .and_then(|v| v.parse().ok());
    let dump0 = std::env::var("BARK3_DUMP_L0").is_ok();
    let dl = dump_layer_env.or(if dump0 { Some(0) } else { dump_layer });

    // Default: the whole conv front end runs on the GPU and its (T, 4096) output
    // stays in VRAM for the encoder. BARK3_CPU_SUB=1 keeps the CPU reference path.
    let (out, t_real, t_sub, t_enc) = if cpu_path {
        let cpu = cpu
            .as_ref()
            .context("CPU backend selected but the CPU encoder is missing")?;
        let sub_rows = subsample(sub_w, &feats.frames, feats.n_valid);
        let t = sub_rows.len() / 4096;
        let t_sub = t1.elapsed().as_secs_f32();
        let t2 = crate::clock::now();
        let pooler = cpu.encode(&sub_rows, t, 24);
        let t_enc = t2.elapsed().as_secs_f32();
        let out = EncodeOut {
            pooler,
            pooler_buf: None,
            block: None,
        };
        (out, t, t_sub, t_enc)
    } else if std::env::var("BARK3_CPU_SUB").is_ok() {
        let enc = &gpu.as_ref().context("wgpu backend missing")?.enc;
        let sub_rows = subsample(sub_w, &feats.frames, feats.n_valid);
        let t = sub_rows.len() / 4096;
        if std::env::var("BARK3_DUMP_SUB").is_ok() {
            std::fs::write("data/dump_sub_cpu.bin", bytemuck::cast_slice(&sub_rows)).ok();
        }
        let t_sub = t1.elapsed().as_secs_f32();
        let t2 = crate::clock::now();
        let out = enc.encode(&sub_rows, t, dl)?;
        (out, t, t_sub, t2.elapsed().as_secs_f32())
    } else {
        let gpu = gpu.as_ref().context("wgpu backend missing")?;
        let (enc, sub_gpu) = (&gpu.enc, &gpu.sub);
        let (buf, t) = sub_gpu.run(&enc.gpu, &feats.frames, feats.n_frames, feats.n_valid)?;
        enc.gpu.flush()?;
        let want_pool = !use_gpu_decoder || dump_pool;
        if std::env::var("BARK3_DUMP_SUB").is_ok() {
            let bytes = enc
                .gpu
                .readback(&buf, (crate::encoder::pad128(t) * 4096 * 4) as u64)?;
            std::fs::write("data/dump_sub_gpu.bin", &bytes).ok();
            std::fs::write("data/dump_mel.bin", bytemuck::cast_slice(&feats.frames)).ok();
            eprintln!(
                "[dbg] dumped gpu sub t={t} frames={} valid={}",
                feats.n_frames, feats.n_valid
            );
        }
        let t_sub = t1.elapsed().as_secs_f32();
        let t2 = crate::clock::now();
        let out = enc.encode_pre(&buf, t, dl, want_pool)?;
        if use_gpu_decoder && !dump_pool {
            // The GPU decoder waits on this queue anyway; drain it here so the
            // encoder and decoder stage timings stay attributable.
            enc.gpu.flush()?;
        }
        (out, t, t_sub, t2.elapsed().as_secs_f32())
    };

    if std::env::var("BARK3_MEMINFO").is_ok() {
        let bytes = gpu.as_ref().map_or(0, |g| g.enc.gpu.allocated_bytes());
        eprintln!(
            "[mem] gpu buffers {:.1} MiB after front end",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }

    if dump_pool {
        for f in [0usize, 500, 1000, 1100, 1127] {
            if (f + 1) * 640 <= out.pooler.len() {
                let row = &out.pooler[f * 640..(f + 1) * 640];
                let mx = row.iter().fold(0f32, |a, v| a.max(v.abs()));
                eprintln!("[pool] frame {f:>5}: max={mx:8.4} first3={:?}", &row[..3]);
            }
        }
        std::fs::write(
            "data/dump_pooler_f32.bin",
            bytemuck::cast_slice(&out.pooler),
        )
        .ok();
        return Ok(TranscribeResult::default());
    }

    if dump0 || dump_layer_env.is_some() {
        if let Some(b) = &out.block {
            // Opt-in file dump so two runs can be compared byte-for-byte.
            let layer = dl.unwrap_or(0);
            std::fs::write(
                format!("data/dump_block_l{layer}.bin"),
                bytemuck::cast_slice(b),
            )
            .ok();
            for f in [0usize, 100, 500, 1000, 1100, 1130, 1200, 1400, t_real - 1] {
                if f < t_real {
                    let row = &b[f * 1024..(f + 1) * 1024];
                    let mx = row.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let nan = row.iter().filter(|v| v.is_nan()).count();
                    eprintln!(
                        "[l0] frame {f:>5}: max={mx:10.3} nan={nan} first3={:?}",
                        &row[..3]
                    );
                }
            }
            return Ok(TranscribeResult::default());
        }
    }

    if !out.pooler.is_empty() {
        let nan = out.pooler.iter().filter(|v| v.is_nan()).count();
        let inf = out.pooler.iter().filter(|v| v.is_infinite()).count();
        if nan > 0 || inf > 0 {
            eprintln!("[warn] pooler nan={nan} inf={inf}/{}", out.pooler.len());
            let first_nan = out
                .pooler
                .chunks(640)
                .position(|c| c.iter().any(|v| v.is_nan()));
            eprintln!("[warn] first NaN frame = {first_nan:?} (t_len {t_real})");
        }
        let frames: Vec<usize> = out
            .pooler
            .chunks(640)
            .enumerate()
            .filter(|(_, c)| c.iter().all(|v| v.abs() < 1e-6))
            .map(|(i, _)| i)
            .collect();
        eprintln!("[dbg] pooler all-zero frames: {}/{}", frames.len(), t_real);
        if !frames.is_empty() {
            eprintln!(
                "[dbg] first zero frames: {:?}",
                &frames[..10.min(frames.len())]
            );
        }
    }

    let t3 = crate::clock::now();
    let tokens = if use_gpu_decoder {
        let gpu = gpu.as_ref().context("wgpu backend missing")?;
        let pooler = out
            .pooler_buf
            .as_ref()
            .context("GPU decoder requested but pooler buffer is missing")?;
        let decoded = gpu.decode.decode(&gpu.enc.gpu, pooler, t_real)?;
        if std::env::var("BARK3_GPUDEC_STATS").is_ok() {
            eprintln!(
                "[gpu-dec] steps={} frame={} chunks={} exit={} record={:.1}ms wait={:.1}ms total={:.1}ms",
                decoded.stats.steps,
                decoded.stats.final_frame,
                decoded.stats.chunks,
                decoded.stats.exit_code,
                decoded.stats.record_ms,
                decoded.stats.wait_ms,
                decoded.stats.total_ms,
            );
        }
        decoded.tokens
    } else {
        tdt.decode(&out.pooler, t_real)
    };
    let t_dec = t3.elapsed().as_secs_f32();
    if std::env::var("BARK3_TOKDUMP").is_ok() {
        let ids: Vec<String> = tokens.iter().map(|t| t.id.to_string()).collect();
        let frames: Vec<String> = tokens.iter().map(|t| t.frame.to_string()).collect();
        let durs: Vec<String> = tokens.iter().map(|t| t.dur.to_string()).collect();
        eprintln!("[tokdump] ids={}", ids.join(","));
        eprintln!("[tokdump] frames={}", frames.join(","));
        eprintln!("[tokdump] durs={}", durs.join(","));
    }
    let transcript = TranscribeResult::new(tokens, vocab(), audio_seconds, &options.segments);
    if std::env::var("BARK3_MEMINFO").is_ok() {
        // Peak of our own GPU buffers (see Gpu::allocated_bytes); zero on the
        // CPU backend, which never creates an adapter.
        let bytes = gpu.as_ref().map_or(0, |g| g.enc.gpu.allocated_bytes());
        eprintln!(
            "[mem] gpu buffers {:.1} MiB",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
    // The processed length includes the 0.25 s pad, which is what the stages
    // above actually ran on.
    let processed = wave.len() as f32 / 16_000.0;
    eprintln!(
        "[time] mel {t_mel:.3}s sub {t_sub:.3}s enc {t_enc:.3}s dec {t_dec:.3}s | audio {processed:.2}s RTFx={:.1}x | tokens {} words {} segments {}",
        processed / (t_mel + t_sub + t_enc + t_dec),
        transcript.tokens.len(),
        transcript.words.len(),
        transcript.segments.len()
    );
    Ok(transcript)
}
