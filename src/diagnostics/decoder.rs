use anyhow::{Context, Result, bail};
use bark3_pro_wgpu::cfg;
use bark3_pro_wgpu::tdt::TdtDecoder;
use bark3_pro_wgpu::transcript::tokens_to_text;

use super::read_f32;
use bark3_pro_wgpu::AsrInference;
use bark3_pro_wgpu::transcript::vocab;

pub fn lstm_check(tdt: &TdtDecoder) -> Result<()> {
    // Feed blank embeddings followed by one real token; print checksums per step.
    let seq = [8192u32; 10]
        .iter()
        .copied()
        .chain([309u32])
        .collect::<Vec<_>>();
    let mut h = vec![vec![0f32; 640]; 2];
    let mut c = vec![vec![0f32; 640]; 2];
    for token in seq {
        tdt.lstm_step(
            &tdt.embed[token as usize * 640..token as usize * 640 + 640],
            &mut h,
            &mut c,
        );
        let dec = tdt.project(&h[1]);
        let dec_sum: f32 = dec.iter().sum();
        let h_sum: f32 = h.iter().map(|v| v.iter().sum::<f32>()).sum();
        println!("[lstm] tok={token} decsum={dec_sum:.6} hsum={h_sum:.6}");
    }
    Ok(())
}

pub fn pool_decode_test(tdt: &TdtDecoder, golden: &std::path::Path) -> Result<()> {
    let pool = read_f32(&golden.join("pooler_out.bin"));
    let t_len = pool.len() / 640;
    let tokens = tdt.decode(&pool, t_len);
    let text = tokens_to_text(&tokens, vocab());
    let want: String = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        golden.join("meta.json"),
    )?)?["text"]["text"]
        .as_str()
        .context("text")
        .map(|s| s.to_string())?;

    println!("[pool-decode] rust TDT on TORCH pooler:");
    println!("{text}");
    println!(
        "[pool-decode] gate: {}",
        if text == want { "MATCH" } else { "MISMATCH" }
    );

    // Step-exact regression against torch's own greedy trace: sequences/durations.
    let raw = std::fs::read(golden.join("sequences.bin"))?;
    let seq: Vec<i64> = raw
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let rawd = std::fs::read(golden.join("durations.bin"))?;
    let dur: Vec<i64> = rawd
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut want_ids = Vec::new();
    let mut want_frames = Vec::new();
    let mut want_durs = Vec::new();
    let mut frame = 0i64;
    for i in 1..seq.len() {
        if seq[i] != cfg::BLANK as i64 {
            want_ids.push(seq[i] as u32);
            want_frames.push(frame as u32);
            want_durs.push(dur[i] as u32);
        }
        frame += dur[i];
    }

    let got_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let got_frames: Vec<u32> = tokens.iter().map(|t| t.frame).collect();
    let got_durs: Vec<u32> = tokens.iter().map(|t| t.dur).collect();
    println!(
        "[pool-decode] torch steps={} tokens={} final_frame={frame} | rust tokens={}",
        seq.len() - 1,
        want_ids.len(),
        tokens.len()
    );
    let ids_ok = got_ids == want_ids;
    let frames_ok = got_frames == want_frames;
    let durs_ok = got_durs == want_durs;
    if !ids_ok {
        for (i, (got, want)) in got_ids.iter().zip(&want_ids).enumerate() {
            if got != want {
                println!("[pool-decode] token mismatch at {i}: rust={got} torch={want}");
                break;
            }
        }
        println!(
            "[pool-decode] token ids: MISMATCH (len rust {} torch {})",
            got_ids.len(),
            want_ids.len()
        );
    }
    if !frames_ok {
        for (i, (got, want)) in got_frames.iter().zip(&want_frames).enumerate() {
            if got != want {
                println!("[pool-decode] frame mismatch at {i}: rust={got} torch={want}");
                break;
            }
        }
        println!("[pool-decode] token frames: MISMATCH");
    }
    if !durs_ok {
        for (i, (got, want)) in got_durs.iter().zip(&want_durs).enumerate() {
            if got != want {
                println!("[pool-decode] duration mismatch at {i}: rust={got} torch={want}");
                break;
            }
        }
        println!("[pool-decode] token durations: MISMATCH");
    }
    println!(
        "[pool-decode] step-exact: {}",
        if ids_ok && frames_ok && durs_ok {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    if !(ids_ok && frames_ok && durs_ok) {
        bail!("pool-decode step mismatch");
    }
    Ok(())
}

pub fn gpu_pool_decode_test(asr: &AsrInference, golden: &std::path::Path) -> Result<()> {
    let pool_host = read_f32(&golden.join("pooler_out.bin"));
    let t_len = pool_host.len() / 640;
    let gpu = &asr.gpu()?.enc.gpu;
    let pool = gpu.storage("torch_pooler", pool_host.len() as u64 * 4);
    gpu.upload(&pool, bytemuck::cast_slice(&pool_host));

    let started = std::time::Instant::now();
    let decoded = asr.gpu()?.decode.decode(gpu, &pool, t_len)?;
    let gpu_time = started.elapsed().as_secs_f32();

    let started = std::time::Instant::now();
    let cpu_tokens = asr.tdt.decode(&pool_host, t_len);
    let cpu_time = started.elapsed().as_secs_f32();

    let gpu_ids: Vec<u32> = decoded.tokens.iter().map(|token| token.id).collect();
    let gpu_frames: Vec<u32> = decoded.tokens.iter().map(|token| token.frame).collect();
    let gpu_durs: Vec<u32> = decoded.tokens.iter().map(|token| token.dur).collect();
    let cpu_ids: Vec<u32> = cpu_tokens.iter().map(|token| token.id).collect();
    let cpu_frames: Vec<u32> = cpu_tokens.iter().map(|token| token.frame).collect();
    let cpu_durs: Vec<u32> = cpu_tokens.iter().map(|token| token.dur).collect();
    let ids_ok = gpu_ids == cpu_ids;
    let frames_ok = gpu_frames == cpu_frames;
    let durs_ok = gpu_durs == cpu_durs;

    println!(
        "[gpu-decode] gpu={gpu_time:.4}s cpu={cpu_time:.4}s steps={} frame={} chunks={} exit={} record={:.1}ms wait={:.1}ms",
        decoded.stats.steps,
        decoded.stats.final_frame,
        decoded.stats.chunks,
        decoded.stats.exit_code,
        decoded.stats.record_ms,
        decoded.stats.wait_ms,
    );
    println!(
        "[gpu-decode] tokens gpu={} cpu={} ids={} frames={} durations={}",
        gpu_ids.len(),
        cpu_ids.len(),
        if ids_ok { "MATCH" } else { "MISMATCH" },
        if frames_ok { "MATCH" } else { "MISMATCH" },
        if durs_ok { "MATCH" } else { "MISMATCH" },
    );
    if !ids_ok {
        for (index, (gpu_id, cpu_id)) in gpu_ids.iter().zip(&cpu_ids).enumerate() {
            if gpu_id != cpu_id {
                println!("[gpu-decode] first id mismatch at {index}: gpu={gpu_id} cpu={cpu_id}");
                break;
            }
        }
    }
    if !frames_ok {
        for (index, (gpu_frame, cpu_frame)) in gpu_frames.iter().zip(&cpu_frames).enumerate() {
            if gpu_frame != cpu_frame {
                println!(
                    "[gpu-decode] first frame mismatch at {index}: gpu={gpu_frame} cpu={cpu_frame}"
                );
                break;
            }
        }
    }

    let gpu_text = tokens_to_text(&decoded.tokens, vocab());
    let cpu_text = tokens_to_text(&cpu_tokens, vocab());
    let reference: String = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        golden.join("meta.json"),
    )?)?["text"]["text"]
        .as_str()
        .context("text")?
        .to_string();
    println!(
        "[gpu-decode] text gpu==cpu: {} | gpu==torch: {}",
        if gpu_text == cpu_text {
            "MATCH"
        } else {
            "MISMATCH"
        },
        if gpu_text == reference {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    if !(ids_ok && frames_ok && durs_ok && gpu_text == reference) {
        bail!("GPU TDT decode is not step-exact against the CPU reference");
    }
    Ok(())
}
