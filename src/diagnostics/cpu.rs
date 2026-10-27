//! CPU encoder backend checks: block-level alignment against the GPU encoder
//! and an end-to-end transcript gate on the golden fixture.

use anyhow::{Context, Result};
use bark3_pro_wgpu::AsrInference;
use bark3_pro_wgpu::sub::subsample;
use bark3_pro_wgpu::transcript::tokens_to_text;

use super::{diff_report, read_f32};
use bark3_pro_wgpu::transcript::vocab;

/// Compare the two backends on an arbitrary wav, stage by stage.
///
/// The golden fixture only covers one utterance length; this reaches the shapes
/// the mini-set exercises, where a length-dependent bug would hide.
pub fn cpu_wav_diff(asr: &AsrInference, wav: &std::path::Path) -> Result<()> {
    let wave = bark3_pro_wgpu::mel::read_wav(wav)?;
    let mut padded = vec![0f32; 4000];
    padded.extend_from_slice(&wave);
    let feats = asr.mel_fe.compute(&padded);
    let sub_rows = subsample(&asr.sub_w, &feats.frames, feats.n_valid);
    let t = sub_rows.len() / 4096;
    let cpu = asr
        .cpu
        .as_ref()
        .context("CPU encoder not loaded (run through Backend::Cpu)")?;
    let gpu = asr.gpu()?.enc.encode(&sub_rows, t, Some(0))?;
    let gpu_block = gpu.block.as_ref().context("GPU layer dump missing")?;
    println!("[cpudiff] {} t={t}", wav.display());
    let h1 = cpu.encode(&sub_rows, t, 1);
    diff_report("h_l0(cpu vs gpu)", &h1[..t * 1024], gpu_block);
    let pooler = cpu.encode(&sub_rows, t, 24);
    let gpu_pooler = asr.gpu()?.enc.encode(&sub_rows, t, None)?;
    diff_report("pooler(cpu vs gpu)", &pooler, &gpu_pooler.pooler);
    Ok(())
}

/// Run the golden fixture through the CPU encoder.
///
/// First one block against the GPU's layer-0 dump (algorithmic check), then the
/// full stack against torch's pooler and the reference transcript.
pub fn cpu_encoder_test(asr: &AsrInference, golden: &std::path::Path) -> Result<()> {
    let wave = read_f32(&golden.join("wave.bin"));
    let g_text: String = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        golden.join("meta.json"),
    )?)?["text"]["text"]
        .as_str()
        .context("text")?
        .to_string();

    let feats = asr.mel_fe.compute(&wave);
    let sub_rows = subsample(&asr.sub_w, &feats.frames, feats.n_valid);
    let t = sub_rows.len() / 4096;
    let cpu = asr
        .cpu
        .as_ref()
        .context("CPU encoder not loaded (run through Backend::Cpu)")?;
    println!("[cpu-enc] T = {t}");

    // One block against the GPU encoder's layer-0 residual stream.
    let gpu = asr.gpu()?.enc.encode(&sub_rows, t, Some(0))?;
    let gpu_block = gpu.block.as_ref().context("GPU layer dump missing")?;
    let h1 = cpu.encode(&sub_rows, t, 1);
    diff_report("h_l0(cpu vs gpu)", &h1[..t * 1024], gpu_block);

    // Full stack.
    let started = std::time::Instant::now();
    let pooler = cpu.encode(&sub_rows, t, 24);
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "[cpu-enc] {t} frames in {seconds:.3}s ({:.2} ms/frame)",
        seconds * 1000.0 / t as f64
    );

    let reference = read_f32(&golden.join("pooler_out.bin"));
    diff_report("pooler(cpu vs torch)", &pooler, &reference);
    if let Ok(gpu_state) = asr.gpu()
        && let Ok(gpu_pooler) = gpu_state.enc.encode(&sub_rows, t, None)
    {
        diff_report("pooler(cpu vs gpu)", &pooler, &gpu_pooler.pooler);
    }

    let tokens = asr.tdt.decode(&pooler, t);
    let text = tokens_to_text(&tokens, vocab());
    println!(
        "[cpu-enc] tokens {} | text ({} chars):",
        tokens.len(),
        text.len()
    );
    println!("{}", &text[..text.len().min(300)]);
    let ok = text == g_text;
    println!(
        "[cpu-enc] transcript gate: {}",
        if ok { "MATCH" } else { "MISMATCH" }
    );
    if !ok {
        println!("[cpu-enc] want: {}", &g_text[..g_text.len().min(300)]);
        anyhow::bail!("cpu encoder transcript mismatch");
    }
    Ok(())
}
