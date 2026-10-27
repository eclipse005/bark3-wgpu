//! Developer-only numerical checks and microbenchmarks.
//!
//! These commands intentionally share the production [`AsrInference`] resources,
//! but none of their dump/compare logic belongs to the transcription module.

mod attention;
mod cpu;
mod decoder;
mod encoder;
mod gemm;
mod golden;

use std::path::Path;

use anyhow::Result;
use bark3_wgpu::AsrInference;

/// Commands that read the torch reference dumps. Those dumps are local
/// verification assets (a few hundred megabytes) and are not shipped, so the
/// directory is either inside the checkout or named by `BARK3_GOLDEN`.
const NEEDS_GOLDEN: &[&str] = &[
    "f2htest",
    "ff1atest",
    "lnf32test",
    "ffntest",
    "epitest",
    "attn32test",
    "attntest",
    "pooldec",
    "gputdt",
    "golden",
    "cpuenc",
];

pub fn dispatch(command: &str, asr: &AsrInference, root: &Path) -> Option<Result<()>> {
    let golden = std::env::var_os("BARK3_GOLDEN")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("data/golden"));
    if NEEDS_GOLDEN.contains(&command) && !golden.is_dir() {
        return Some(Err(anyhow::anyhow!(
            "`{command}` needs the torch reference dumps, which are not shipped:              they are at {} (set BARK3_GOLDEN to point elsewhere)",
            golden.display()
        )));
    }
    let golden = golden.as_path();
    let result = match command {
        "f2htest" => asr
            .gpu()
            .and_then(|g| encoder::f2h_mix_test(&g.enc, golden)),
        "ff1atest" => asr.gpu().and_then(|g| encoder::ff1a_test(&g.enc, golden)),
        "lnf32test" => asr.gpu().and_then(|g| encoder::lnf32_test(&g.enc, golden)),
        "lntest" => asr.gpu().and_then(|g| encoder::ln_test(&g.enc)),
        "ffntest" => asr.gpu().and_then(|g| encoder::ffn_test(&g.enc, golden)),
        "epitest" => asr.gpu().and_then(|g| encoder::epi_test(&g.enc, golden)),
        "attn32test" => asr
            .gpu()
            .and_then(|g| attention::attn32_test(&g.enc, golden)),
        "attntest" => asr.gpu().and_then(|g| attention::attn_test(&g.enc, golden)),
        "gmixz8" => asr.gpu().and_then(|g| gemm::gemm_mix_z8_test(&g.enc)),
        "gmixtb" => asr.gpu().and_then(|g| gemm::gemm_mix_tb_test(&g.enc)),
        "gmixtest" => asr.gpu().and_then(|g| gemm::gemm_mix_test(&g.enc)),
        "gemmtest" => asr.gpu().and_then(|g| gemm::gemm_test(&g.enc)),
        "lbtest" => asr.gpu().and_then(|g| gemm::lowbit_vs_f16_test(&g.enc)),
        "gemmbench" => asr.gpu().and_then(|g| gemm::gemm_bench(&g.enc)),
        "decbench" => {
            asr.tdt.bench(2000);
            Ok(())
        }
        "lstmchk" => decoder::lstm_check(&asr.tdt),
        "pooldec" => decoder::pool_decode_test(&asr.tdt, golden),
        "gputdt" => decoder::gpu_pool_decode_test(asr, golden),
        "golden" => golden::golden_mode(asr, golden),
        "cpuenc" => cpu::cpu_encoder_test(asr, golden),
        "cpudiff" => match std::env::args().nth(2) {
            Some(wav) => cpu::cpu_wav_diff(asr, Path::new(&wav)),
            None => Err(anyhow::anyhow!("usage: bark3 cpudiff <16k-mono.wav>")),
        },
        _ => return None,
    };
    Some(result)
}

pub fn dump_wgsl() -> Result<()> {
    let wgsl = bark3_wgpu::encoder::gemm_mix(false);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/gemm_mix.wgsl");
    let file = std::fs::File::create(&path)?;
    std::io::Write::write_all(&mut std::io::BufWriter::new(file), wgsl.as_bytes())?;
    println!("wrote {} ({} lines)", path.display(), wgsl.lines().count());
    Ok(())
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff_report(name: &str, got: &[f32], want: &[f32]) -> (f32, f64) {
    let n = got.len().min(want.len());
    let mut max_abs = 0f32;
    let mut mean_abs = 0f64;
    for i in 0..n {
        let d = (got[i] - want[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        mean_abs += d as f64;
    }
    mean_abs /= n.max(1) as f64;
    let nan_got = got.iter().filter(|v| v.is_nan()).count();
    let inf_got = got.iter().filter(|v| v.is_infinite()).count();
    println!(
        "  {name:<18} max|Δ|={max_abs:.4}  mean|Δ|={mean_abs:.5}  (n={n}, nan={nan_got}, inf={inf_got})",
    );
    println!("    got[0..4]  = {:?}", &got[..4.min(got.len())]);
    println!("    want[0..4]= {:?}", &want[..4.min(want.len())]);
    (max_abs, mean_abs)
}
