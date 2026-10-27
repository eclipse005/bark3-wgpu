//! Command-line entry point. Dispatch only: the model, the pipeline and the
//! output renderings live in the library.

mod diagnostics;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use bark3_pro_wgpu::{AsrInference, Backend, TranscribeOptions};

const USAGE: &str = "usage: bark3 --model <checkpoint-dir> transcribe <16k-mono.wav>\n\
                     \x20      bark3 --model <checkpoint-dir> golden | pooldec | cpuenc | <diagnostic>\n\
                     \x20      bark3 dumpwgsl\n\
                     \n\
                     <checkpoint-dir> holds model.safetensors (plus lowbit.json for a compressed\n\
                     Bark 3.0 release). BARK3_MODEL can stand in for --model.";

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    // This diagnostic only generates source text; it does not need a GPU or model.
    if args.first().map(String::as_str) == Some("dumpwgsl") {
        return diagnostics::dump_wgsl();
    }

    // The checkpoint is an input, not something to guess: pass it explicitly or
    // name it in BARK3_MODEL. A wrong directory is a user error, and a silently
    // picked sibling directory is worse than an error.
    let model = take_model(&mut args)?;
    let command = args.first().map(String::as_str).context(USAGE)?;

    if command == "transcribe" {
        let wav = args.get(1).context(USAGE)?;
        if let Some(extra) = args.get(2) {
            bail!("unknown argument `{extra}`\n{USAGE}");
        }
        let asr = AsrInference::load(&model, backend())?;
        let result = asr.transcribe(wav, TranscribeOptions::default())?;
        println!("{}", result.to_json());
        return Ok(());
    }

    // `cpuenc` / `cpudiff` compare the two encoders, so they need both loaded.
    let backend = if command.starts_with("cpu") {
        Backend::Gpu
    } else {
        backend()
    };
    let mut asr = AsrInference::load(&model, backend)?;
    if command.starts_with("cpu") {
        asr.load_cpu_encoder()?;
    }
    if let Some(result) = diagnostics::dispatch(command, &asr, root) {
        return result;
    }

    bail!("unknown command `{command}`\n{USAGE}")
}

/// `--model <dir>` (anywhere in the argument list) or `BARK3_MODEL`, and nothing
/// else: the binary ships no default checkpoint path.
fn take_model(args: &mut Vec<String>) -> Result<PathBuf> {
    let flag = args.iter().position(|a| a == "--model");
    let from_flag = match flag {
        Some(i) => {
            if i + 1 >= args.len() {
                bail!("--model needs a directory\n{USAGE}");
            }
            let dir = args.remove(i + 1);
            args.remove(i);
            Some(PathBuf::from(dir))
        }
        None => None,
    };
    let dir = match from_flag.or_else(|| std::env::var_os("BARK3_MODEL").map(PathBuf::from)) {
        Some(dir) => dir,
        None => {
            bail!("no checkpoint given: pass --model <checkpoint-dir> or set BARK3_MODEL\n{USAGE}")
        }
    };
    anyhow::ensure!(
        dir.join("model.safetensors").is_file(),
        "{} does not contain model.safetensors",
        dir.display()
    );
    Ok(dir)
}

/// The compute path for this process. `BARK3_CPU_ENC=1` is the CPU-only
/// backend; otherwise the best backend wins (GPU when it can be built).
fn backend() -> Backend {
    if std::env::var("BARK3_CPU_ENC").is_ok() {
        Backend::Cpu
    } else {
        Backend::Auto
    }
}
