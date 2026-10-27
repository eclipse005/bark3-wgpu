//! Safetensors weight access for the wgpu crate.
//!
//! [`TensorReader`] hands out one tensor at a time instead of mapping the whole
//! checkpoint.  The mmap variant it replaced kept every touched page resident
//! while the consumer built its own copy: on the CPU backend that meant a 1.25 GB
//! mapping coexisting with 2.4 GB of f32 weights, and on the wgpu backend a
//! 1.9 GB peak that was pure page cache.  Reading a tensor into an owned buffer,
//! using it, and dropping it keeps only one tensor in flight.  The bytes are the
//! same ones the mapping would have produced, so results do not change by a bit.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::raw::RawTensor;
use anyhow::{Context, anyhow, bail};
use bytes::Bytes;

/// One tensor's location inside a checkpoint.
struct Entry {
    /// Index into [`TensorReader::files`].
    file: usize,
    offset: u64,
    len: usize,
    shape: Vec<usize>,
    /// `None` for dtypes the consumers never ask for (`num_batches_tracked` is
    /// I64); the error surfaces only if such a tensor is actually requested.
    dtype: Option<safetensors::Dtype>,
}

pub struct TensorReader {
    files: Vec<RefCell<File>>,
    entries: HashMap<String, Entry>,
}

impl TensorReader {
    /// Opens `<model_dir>/model.safetensors`, following
    /// `model.safetensors.index.json` when the checkpoint is sharded.
    pub fn open_model(model_dir: &Path) -> anyhow::Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Self::open(&model_dir.join("model.safetensors"));
        }
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid index.json"))?;
        let shards: BTreeSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
        let mut files = Vec::with_capacity(shards.len());
        let mut entries = HashMap::new();
        for (i, shard) in shards.iter().enumerate() {
            let (e, file) = read_shard(&model_dir.join(shard), i)?;
            entries.extend(e);
            files.push(RefCell::new(file));
        }
        Ok(Self { files, entries })
    }

    /// Opens a single-file checkpoint.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let (entries, file) = read_shard(path, 0)?;
        Ok(Self {
            files: vec![RefCell::new(file)],
            entries,
        })
    }

    /// Reads one tensor into an owned buffer; the caller drops it once the
    /// consumer has copied what it needs.
    ///
    /// Takes `&self` (the file cursors sit behind `RefCell`s) so loaders can
    /// accept it as an ordinary `Fn`, exactly like a weight map.
    pub fn read(&self, name: &str) -> anyhow::Result<RawTensor> {
        let entry = self
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let mut data = vec![0u8; entry.len];
        let mut file = self.files[entry.file].borrow_mut();
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| anyhow!("seek {name}: {e}"))?;
        file.read_exact(&mut data)
            .map_err(|e| anyhow!("read {name}: {e}"))?;
        drop(file);
        Ok(RawTensor {
            data: Bytes::from(data),
            shape: entry.shape.clone(),
            dtype: entry
                .dtype
                .with_context(|| format!("{name}: unsupported dtype"))?,
        })
    }
}

/// What a checkpoint directory holds, read from its header alone.
///
/// A model directory is self-describing enough to answer "what did you give me?":
/// the dtype mix says which release family it is (`bf16` is the full-precision
/// one, `u32` codes plus a `bark3.json` is a compressed release), and the tensor
/// names carry the architecture. [`ModelInfo::check_architecture`] turns that
/// into a refusal to run a checkpoint this build was not compiled for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Release name (e.g. `Underdog-Bark-3.0-Plus`) when `bark3.json` records it.
    /// A bare safetensors directory has no name to read, only a layout.
    pub release: Option<String>,
    pub layout: Layout,
    pub tensors: usize,
    /// Logical weights: a packed 4-bit code word counts as its eight weights.
    pub weights: u64,
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Bf16,
    F16,
    F32,
    /// 4-bit codes + per-row fp16 levels (`lowbit.json` lists the level counts).
    LowBit,
}

impl Layout {
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Layout::Bf16 => "bf16",
            Layout::F16 => "f16",
            Layout::F32 => "f32",
            Layout::LowBit => "low-bit",
        }
    }
}

impl ModelInfo {
    /// Refuse a checkpoint whose architecture differs from the compiled-in
    /// constants, naming both sides instead of failing on a missing tensor later.
    pub fn check_architecture(&self) -> anyhow::Result<()> {
        use crate::cfg;
        let expected = (cfg::LAYERS, cfg::HIDDEN, cfg::FFN, cfg::VOCAB);
        if (self.layers, self.hidden, self.ffn, self.vocab) == expected {
            return Ok(());
        }
        bail!(
            "this checkpoint is {}x{} (ffn {}, vocab {}), but this build is \
             compiled for {}x{} (ffn {}, vocab {}): it is a different Parakeet \
             variant, not a Bark 3.0 release",
            self.layers,
            self.hidden,
            self.ffn,
            self.vocab,
            expected.0,
            expected.1,
            expected.2,
            expected.3,
        )
    }

    /// One-line summary for the load log.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} ({}, {:.0}M weights, {} tensors)",
            self.release.as_deref().unwrap_or("unnamed checkpoint"),
            self.layout.tag(),
            self.weights as f64 / 1e6,
            self.tensors
        )
    }
}

/// Reads `<model_dir>/model.safetensors`'s header (not the weights) and
/// classifies it. Cheap: a few hundred kilobytes of JSON.
pub fn probe(model_dir: &Path) -> anyhow::Result<ModelInfo> {
    let path = model_dir.join("model.safetensors");
    let (entries, _file) = read_shard(&path, 0)?;

    let mut lowbit = false;
    let mut dtypes = std::collections::BTreeSet::new();
    let mut weights = 0u64;
    for (name, e) in &entries {
        if name.ends_with(".levels") {
            continue; // level tables are per-row metadata, not weights
        }
        let dtype = match e.dtype {
            Some(d) => d,
            None => continue,
        };
        dtypes.insert(format!("{dtype:?}"));
        let numel: u64 = e.shape.iter().map(|d| *d as u64).product();
        weights += match dtype {
            safetensors::Dtype::U32 => {
                lowbit = true;
                numel * 8
            }
            _ => numel,
        };
    }
    if dtypes.contains("U32") && !model_dir.join("lowbit.json").is_file() {
        bail!(
            "{}: weights are stored as 4-bit codes but lowbit.json is missing; \
             reconvert the release with bark_tools/encode_lowbit.py",
            path.display()
        );
    }
    if model_dir.join("lowbit.json").is_file() {
        lowbit = true;
    }

    let layout = if lowbit {
        Layout::LowBit
    } else if dtypes.contains("BF16") {
        Layout::Bf16
    } else if dtypes.contains("F16") {
        Layout::F16
    } else if dtypes.contains("F32") {
        Layout::F32
    } else {
        bail!("{}: no float or code tensors found", path.display());
    };

    // Architecture from shapes: a 4-bit code row packs eight weights per word.
    let shape_of =
        |name: &str| -> Option<Vec<usize>> { entries.get(name).map(|e| e.shape.clone()) };
    let ffn_w = shape_of("encoder.layers.0.feed_forward1.linear1.weight").ok_or_else(|| {
        anyhow!(
            "{}: not a Parakeet TDT checkpoint (no feed_forward1)",
            path.display()
        )
    })?;
    let ffn = ffn_w[0];
    let hidden = if lowbit { ffn_w[1] * 8 } else { ffn_w[1] };
    let vocab = shape_of("decoder.embedding.weight").map_or(0, |s| s[0]);
    let mut layers = std::collections::BTreeSet::new();
    for name in entries.keys() {
        if let Some(rest) = name.strip_prefix("encoder.layers.") {
            if let Some((n, _)) = rest.split_once('.') {
                if let Ok(n) = n.parse::<usize>() {
                    layers.insert(n);
                }
            }
        }
    }
    let release = std::fs::read_to_string(model_dir.join("bark3.json"))
        .ok()
        .and_then(|text| {
            serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["source"].as_str().map(str::to_string))
        });

    Ok(ModelInfo {
        release,
        layout,
        tensors: entries.len(),
        weights,
        layers: layers.len(),
        hidden,
        ffn,
        vocab,
    })
}

/// Parses one safetensors header and returns its tensors plus the open file.
fn read_shard(path: &Path, index: usize) -> anyhow::Result<(HashMap<String, Entry>, File)> {
    let mut file = File::open(path).map_err(|e| anyhow!("open {:?}: {}", path, e))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| anyhow!("read header length: {}", e))?;
    let header_len = u64::from_le_bytes(len_buf);
    // The header is JSON describing shapes; anything bigger is a wrong file.
    if header_len > 64 << 20 {
        bail!(
            "{}: safetensors header claims {header_len} bytes",
            path.display()
        );
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|e| anyhow!("read header: {}", e))?;
    let json: serde_json::Value = serde_json::from_slice(&header)?;
    let base = 8 + header_len;
    let mut entries = HashMap::new();
    for (name, meta) in json.as_object().context("safetensors header")? {
        if name == "__metadata__" {
            continue;
        }
        let shape = meta["shape"]
            .as_array()
            .with_context(|| format!("{name}: shape"))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        let offsets = meta["data_offsets"]
            .as_array()
            .with_context(|| format!("{name}: data_offsets"))?;
        let begin = offsets[0]
            .as_u64()
            .with_context(|| format!("{name}: begin"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("{name}: end"))?;
        let dtype = match meta["dtype"].as_str().unwrap_or("") {
            "BF16" => Some(safetensors::Dtype::BF16),
            "F16" => Some(safetensors::Dtype::F16),
            "F32" => Some(safetensors::Dtype::F32),
            "U32" => Some(safetensors::Dtype::U32),
            _ => None,
        };
        entries.insert(
            name.clone(),
            Entry {
                file: index,
                offset: base + begin,
                len: (end - begin) as usize,
                shape,
                dtype,
            },
        );
    }
    Ok((entries, file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;

    fn info(layers: usize, hidden: usize, ffn: usize, vocab: usize) -> ModelInfo {
        ModelInfo {
            release: None,
            layout: Layout::Bf16,
            tensors: 723,
            weights: 627_057_286,
            layers,
            hidden,
            ffn,
            vocab,
        }
    }

    /// The shapes this build was compiled for are accepted; anything else is
    /// refused by name, before a gigabyte goes to the GPU.
    #[test]
    fn architecture_guard_accepts_this_model_and_rejects_others() {
        assert!(
            info(cfg::LAYERS, cfg::HIDDEN, cfg::FFN, cfg::VOCAB)
                .check_architecture()
                .is_ok()
        );
        let err = info(12, 768, 3072, 1025)
            .check_architecture()
            .unwrap_err()
            .to_string();
        assert!(err.contains("12x768"), "{err}");
        assert!(err.contains("1025"), "{err}");
    }
}
