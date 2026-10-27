//! Weight sources: one interface over the two layouts the official releases ship.
//!
//! * Bark 3.0 Pro is a `model.safetensors` checkpoint (bf16).
//! * Bark 3.0 / 3.0 Plus / 3.0 Clean are a `model.bark` container whose encoder
//!   linears keep 4-bit codes and per-row fp16 levels.
//!
//! Both are consumed as downloaded — no conversion step. A source can hand the
//! GEMM either f16 values ([`WeightSource::read`]) or the low-bit pair
//! ([`WeightSource::read_lowbit`]); the latter is what the low-bit kernel wants
//! and what keeps a 163 MB release at ~380 MiB of VRAM.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::bark::{BarkContainer, LowBitTensor};
use crate::raw::RawTensor;
use crate::weights_io::{ModelInfo, TensorReader, probe};

/// Any checkpoint this crate can load.
pub trait WeightSource {
    /// Tensor as floating-point (or fp16) values.
    fn read(&self, name: &str) -> Result<RawTensor>;
    /// The 4-bit form, when the checkpoint stores this tensor that way.
    fn read_lowbit(&self, name: &str) -> Result<Option<Arc<LowBitTensor>>> {
        let _ = name;
        Ok(None)
    }
    /// Optional warm-up (parallel decode for the container format).
    fn prefetch(&self) -> Result<()> {
        Ok(())
    }
    /// What this source holds, read from its header.
    fn info(&self) -> &ModelInfo;
}

/// A `model.safetensors` checkpoint (the full-precision release).
pub struct SafetensorsSource {
    reader: TensorReader,
    info: ModelInfo,
}

impl SafetensorsSource {
    fn open(dir: &Path) -> Result<Self> {
        let info = probe(dir)?;
        let reader = TensorReader::open_model(dir)?;
        Ok(Self { reader, info })
    }
}

impl WeightSource for SafetensorsSource {
    fn read(&self, name: &str) -> Result<RawTensor> {
        self.reader.read(name)
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }
}

impl WeightSource for BarkContainer {
    fn read(&self, name: &str) -> Result<RawTensor> {
        BarkContainer::read(self, name)
    }

    fn read_lowbit(&self, name: &str) -> Result<Option<Arc<LowBitTensor>>> {
        BarkContainer::read_lowbit(self, name)
    }

    fn prefetch(&self) -> Result<()> {
        BarkContainer::prefetch(self)
    }

    fn info(&self) -> &ModelInfo {
        BarkContainer::info(self)
    }
}

/// Opens the checkpoint in `dir`, whichever of the two layouts it holds.
pub fn open(dir: &Path) -> Result<Box<dyn WeightSource>> {
    let bark = dir.join("model.bark");
    let safetensors = dir.join("model.safetensors");
    if bark.is_file() {
        Ok(Box::new(BarkContainer::open(dir)?))
    } else if safetensors.is_file() {
        Ok(Box::new(SafetensorsSource::open(dir)?))
    } else {
        bail!(
            "{} has neither model.bark nor model.safetensors",
            dir.display()
        )
    }
}

/// Build a source over an in-memory checkpoint map (tests and diagnostics).
pub fn from_map(map: &std::collections::HashMap<String, RawTensor>) -> impl WeightSource + '_ {
    struct Map<'a>(&'a std::collections::HashMap<String, RawTensor>, ModelInfo);
    impl WeightSource for Map<'_> {
        fn read(&self, name: &str) -> Result<RawTensor> {
            self.0
                .get(name)
                .cloned()
                .with_context(|| format!("missing tensor {name}"))
        }
        fn info(&self) -> &ModelInfo {
            &self.1
        }
    }
    Map(
        map,
        ModelInfo {
            release: None,
            layout: crate::weights_io::Layout::Bf16,
            tensors: map.len(),
            weights: 0,
            layers: crate::cfg::LAYERS,
            hidden: crate::cfg::HIDDEN,
            ffn: crate::cfg::FFN,
            vocab: crate::cfg::VOCAB,
        },
    )
}
