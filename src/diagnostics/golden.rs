use anyhow::{Context, Result, bail};
use bark3_wgpu::encoder::pos_embed_f32;
use bark3_wgpu::sub::subsample;
use bark3_wgpu::transcript::{tokens_to_text, vocab};

use bark3_wgpu::AsrInference;

use super::{diff_report, read_f32};

pub fn golden_mode(asr: &AsrInference, golden: &std::path::Path) -> Result<()> {
    let (tdt, sub_w, mel_fe) = (&asr.tdt, &asr.sub_w, &asr.mel_fe);
    let enc = &asr.gpu()?.enc;
    let wave = read_f32(&golden.join("wave.bin"));
    let g_in = read_f32(&golden.join("input_features.bin"));
    let g_pos = read_f32(&golden.join("pos_embed.bin"));
    let g_text: String = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        golden.join("meta.json"),
    )?)?["text"]["text"]
        .as_str()
        .context("text")
        .map(|s| s.to_string())?;

    println!("[golden] wave {} samples", wave.len());

    let feats = mel_fe.compute(&wave);
    println!(
        "[golden] frames {} valid {} (ref 1526/1525)",
        feats.n_frames, feats.n_valid
    );
    diff_report("mel", &feats.frames, &g_in);

    let pos = pos_embed_f32(feats.n_valid.div_ceil(8).max(1));
    diff_report("pos_embed", &pos, &g_pos);

    let sub_rows = subsample(sub_w, &feats.frames, feats.n_valid);
    let t_real = sub_rows.len() / 4096;
    println!("[golden] T = {t_real} (ref 191)");
    if let Ok(stage) = std::env::var("BARK3_STAGE") {
        let out = enc.encode(&sub_rows, t_real, None)?;
        if let Some(b) = out.block {
            let nan = b.iter().filter(|v| v.is_nan()).count();
            let half = b.len() / 2;
            let nan_lo = b[..half].iter().filter(|v| v.is_nan()).count();
            let nan_hi = b[half..].iter().filter(|v| v.is_nan()).count();
            println!(
                "[stage {stage}] h nan={}/{} first={:?} lo={}/{} hi={}/{}",
                nan,
                b.len(),
                &b[..4.min(b.len())],
                nan_lo,
                half,
                nan_hi,
                b.len() - half
            );
            let w = 1024;
            let rows = b.len() / w;
            let nan_rows: Vec<usize> = (0..rows)
                .filter(|&r| b[r * w..(r + 1) * w].iter().any(|v| v.is_nan()))
                .collect();
            println!(
                "    nan_rows={}/{} first_nan_rows={:?}",
                nan_rows.len(),
                rows,
                &nan_rows[..8.min(nan_rows.len())]
            );
            if stage == "sublin" {
                let reference = read_f32(&golden.join("subsampling_out.bin"));
                diff_report("sub_linear", &b, &reference);
            }
            if stage == "ff1" {
                let reference = read_f32(&golden.join("torch_h_ff1.bin"));
                diff_report("h_ff1", &b, &reference);
            }
            if stage == "ch2f" {
                let reference = read_f32(&golden.join("torch_attnout.bin"));
                diff_report("attnout", &b, &reference);
            }
            if stage == "conv" {
                let reference = read_f32(&golden.join("torch_h_after_conv.bin"));
                diff_report("h_after_conv_fp32", &b, &reference);
                let reference16 = read_f32(&golden.join("t16_fp16_h3.bin"));
                diff_report("h_after_conv_fp16", &b, &reference16);
            }
            if stage == "ln2" {
                let reference = read_f32(&golden.join("torch_normed_conv.bin"));
                diff_report("normed_conv_fp32", &b, &reference);
                let reference16 = read_f32(&golden.join("t16_normed_conv.bin"));
                diff_report("normed_conv_fp16", &b, &reference16);
            }
            if stage == "pw1" {
                let reference = read_f32(&golden.join("torch_pw1.bin"));
                diff_report("pw1_real", &b[..b.len().min(t_real * 2048)], &reference);
            }
            if stage == "glu" {
                let reference = read_f32(&golden.join("torch_glu.bin"));
                diff_report("glu_real", &b[..b.len().min(t_real * 1024)], &reference);
            }
            if stage == "dw" {
                let reference = read_f32(&golden.join("torch_dw.bin"));
                diff_report("dw_real", &b[..b.len().min(t_real * 1024)], &reference);
            }
            if stage == "ff2b" {
                let reference = read_f32(&golden.join("torch_ffb0.bin"));
                diff_report("ffb0_real", &b[..b.len().min(t_real * 4096)], &reference);
            }
            if stage == "ln1" {
                let reference = read_f32(&golden.join("torch_normed_sa.bin"));
                diff_report("normed_sa", &b, &reference);
            }
            if stage == "pw1" {
                let reference = read_f32(&golden.join("torch_pw1.bin"));
                diff_report("pw1_vs_fp32", &b[..b.len().min(t_real * 2048)], &reference);
                let reference16 = read_f32(&golden.join("t16_fp16_pw1.bin"));
                diff_report(
                    "pw1_vs_fp16",
                    &b[..b.len().min(t_real * 2048)],
                    &reference16,
                );
            }
            if stage == "glu" {
                let reference = read_f32(&golden.join("torch_glu.bin"));
                diff_report("glu_out", &b, &reference);
            }
            if stage == "dw" {
                let reference = read_f32(&golden.join("torch_dw.bin"));
                diff_report("dw_out", &b, &reference);
            }
            if stage == "qu" {
                let reference = read_f32(&golden.join("a32_qu.bin"));
                diff_report("qu_vs_torch", &b, &reference);
            }
            if stage == "crel" {
                let reference = read_f32(&golden.join("a32_scores.bin"));
                diff_report("scores_vs_torch", &b, &reference);
            }
            if stage == "csm" {
                let reference = read_f32(&golden.join("a32_probs.bin"));
                diff_report("probs_vs_torch", &b, &reference);
            }
        }
        return Ok(());
    }

    // Golden block files hold torch's pre-norm block outputs.
    for (layer, reference_file) in [
        (0usize, "torch_h_b0.bin"),
        (1, "torch_h_b1.bin"),
        (2, "torch_h_b2.bin"),
        (3, "torch_h_b3.bin"),
        (11, "torch_h_b11.bin"),
        (23, "torch_h_b23.bin"),
    ] {
        let out = enc.encode(&sub_rows, t_real, Some(layer))?;
        if let Some(block) = &out.block {
            let reference = read_f32(&golden.join(reference_file));
            diff_report(&format!("h_l{layer}(pre-norm)"), block, &reference);
        }
    }

    let out = enc.encode(&sub_rows, t_real, None)?;
    let reference_pool = read_f32(&golden.join("pooler_out.bin"));
    diff_report("pooler_out", &out.pooler, &reference_pool);

    let tokens = tdt.decode(&out.pooler, t_real);
    let text = tokens_to_text(&tokens, vocab());
    println!(
        "[golden] tokens {} | text ({} chars):",
        tokens.len(),
        text.len()
    );
    println!("{}", &text[..text.len().min(300)]);
    let full_match = text == g_text;
    println!(
        "[golden] transcript gate: {}",
        if full_match { "MATCH" } else { "MISMATCH" }
    );
    if !full_match {
        println!("[golden] want: {}", &g_text[..g_text.len().min(300)]);
        bail!("golden mismatch");
    }
    Ok(())
}
