//! Reader for the official `model.bark` container.
//!
//! A compressed Bark 3.0 release ships one `model.bark` file: a JSON index
//! followed by per-tensor blobs, where the encoder linears are stored as a
//! per-row fp16 level table plus one 4-bit ANS-coded symbol per weight, the
//! remaining matrices as 5/6-bit ANS-coded integers times a per-row scale, and
//! the small tensors as fp16.
//!
//! Decoding is exact — the values here are the values the reference loader
//! produces — and the 4-bit layers stay 4-bit: [`BarkContainer::read_lowbit`]
//! hands the GEMM the code words and the level table directly, so the compressed
//! release needs no conversion step and no 1.2 GB f16 expansion.
//!
//! ANS decoding uses `constriction` with the same preset as the reference
//! (u32 words / 64-bit state, 24-bit categorical models); LZMA is `lzma-rs`.
//! Tensors are independent, so `prefetch` decodes them in parallel.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use constriction::stream::{
    Decode, model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder,
};

use crate::raw::RawTensor;

/// One 4-bit tensor in the layout the GEMM expects.
#[derive(Debug, Clone)]
pub struct LowBitTensor {
    /// `rows * ceil(cols/8)` words, eight 4-bit codes per word along the columns.
    pub codes: Vec<u32>,
    /// `rows * 4` halves: the row's levels, zero padded.
    pub levels: Vec<u16>,
    /// Levels actually in use per row (2 or 3).
    pub k: u32,
    pub rows: usize,
    pub cols: usize,
}

impl LowBitTensor {
    pub fn code_words(&self) -> usize {
        self.rows * self.cols.div_ceil(8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Per-row fp16 levels + 4-bit code per weight.
    Levels2,
    /// Per-row symmetric integers (`bits` wide) times a per-row fp16 scale.
    AnsInt { bits: u32 },
    /// Plain little-endian fp16 / int64 / int32.
    Raw,
}

#[derive(Debug, Clone)]
struct Entry {
    kind: Kind,
    shape: Vec<usize>,
    level_count: u32,
    /// Payload offset inside the file.
    offset: u64,
    lev: u64,
    meta: u64,
    words: u64,
    scale: u64,
    zipped_levels: bool,
    zipped_scale: bool,
    zipped_raw: bool,
    /// `torch.int64` / `torch.int32` for `raw` entries (the rest are fp16).
    raw_dtype: safetensors::Dtype,
}

impl Entry {
    fn n(&self) -> usize {
        self.shape.iter().product()
    }

    fn rows_cols(&self) -> (usize, usize) {
        (self.shape[0], self.n() / self.shape[0].max(1))
    }
}

pub struct BarkContainer {
    source: Source,
    entries: HashMap<String, Entry>,
    /// Decoded 4-bit tensors, filled by [`Self::prefetch`].
    cache: std::sync::Mutex<HashMap<String, std::sync::Arc<LowBitTensor>>>,
    info: crate::weights_io::ModelInfo,
}

impl BarkContainer {
    /// Opens `<dir>/model.bark`.
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("model.bark");
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = file
            .try_clone()
            .with_context(|| format!("clone {}", path.display()))?;
        let header =
            read_header(&mut reader).with_context(|| format!("{}: header", path.display()))?;
        let entries = parse_index(&header, 8 + header.len() as u64)?;
        let info = Self::describe(
            &entries,
            dir.file_name().map(|n| n.to_string_lossy().into_owned()),
        );
        Ok(Self {
            source: Source::File(file),
            entries,
            cache: std::sync::Mutex::new(HashMap::new()),
            info,
        })
    }

    /// Opens a container that is already in memory — the wasm entry point, where
    /// the weights arrive as a fetched `Uint8Array`.
    pub fn from_bytes(data: std::sync::Arc<Vec<u8>>, release: Option<String>) -> Result<Self> {
        let header =
            read_header(&mut std::io::Cursor::new(&data[..])).context("container header")?;
        let entries = parse_index(&header, 8 + header.len() as u64)?;
        let info = Self::describe(&entries, release);
        Ok(Self {
            source: Source::Memory(data),
            entries,
            cache: std::sync::Mutex::new(HashMap::new()),
            info,
        })
    }

    fn describe(
        entries: &HashMap<String, Entry>,
        release: Option<String>,
    ) -> crate::weights_io::ModelInfo {
        let architecture = Self::architecture_of(entries).unwrap_or((0, 0, 0, 0));
        crate::weights_io::ModelInfo {
            release,
            layout: crate::weights_io::Layout::LowBit,
            tensors: entries.len(),
            weights: Self::weight_count_of(entries),
            layers: architecture.0,
            hidden: architecture.1,
            ffn: architecture.2,
            vocab: architecture.3,
        }
    }

    /// What this container holds, read from its header at open time.
    pub fn info(&self) -> &crate::weights_io::ModelInfo {
        &self.info
    }

    /// Logical weights: every entry's element count (4-bit tensors included as
    /// the logical shape the header records).
    fn weight_count_of(entries: &HashMap<String, Entry>) -> u64 {
        entries
            .values()
            .filter(|e| {
                e.raw_dtype != safetensors::Dtype::I64 && e.raw_dtype != safetensors::Dtype::I32
            })
            .map(|e| e.n() as u64)
            .sum()
    }

    /// Tensor names, in container order is not kept; sorted for stable logs.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of encoder layers, from the tensor names.
    pub fn layers(&self) -> usize {
        Self::layers_of(&self.entries)
    }

    fn layers_of(entries: &HashMap<String, Entry>) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for name in entries.keys() {
            if let Some(rest) = name.strip_prefix("encoder.layers.") {
                if let Some((n, _)) = rest.split_once('.') {
                    if let Ok(n) = n.parse::<usize>() {
                        seen.insert(n);
                    }
                }
            }
        }
        seen.len()
    }

    /// Architecture as the container describes it: `(layers, hidden, ffn, vocab)`.
    fn architecture_of(entries: &HashMap<String, Entry>) -> Option<(usize, usize, usize, usize)> {
        let ffn = entries.get("encoder.layers.0.feed_forward1.linear1.weight")?;
        // The container records logical shapes even for the 4-bit tensors, so
        // the weight's second dimension is already the hidden size.
        let (ffn_rows, hidden) = ffn.rows_cols();
        let vocab = entries.get("decoder.embedding.weight")?.shape[0];
        Some((Self::layers_of(entries), hidden, ffn_rows, vocab))
    }

    fn blob(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        match &self.source {
            Source::Memory(bytes) => {
                let start = offset as usize;
                bytes
                    .get(start..start + len as usize)
                    .context("blob outside the buffer")?
                    .to_vec()
                    .pipe(Ok)
            }
            Source::File(file) => {
                // Positional read: `prefetch` reads tensors from several threads,
                // and a shared cursor (or a cloned handle, which shares it on
                // Windows) would hand each of them someone else's bytes.
                let mut buf = vec![0u8; len as usize];
                read_at(file, &mut buf, offset).context("read container blob")?;
                Ok(buf)
            }
        }
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries
            .get(name)
            .with_context(|| format!("{name}: not in the container"))
    }

    /// Bytes of an entry field, decompressed when the container LZMA-compressed it.
    fn field(&self, e: &Entry, offset_key: u64, len: u64, zipped: bool) -> Result<Vec<u8>> {
        let raw = self.blob(e.offset + offset_key, len)?;
        if !zipped {
            return Ok(raw);
        }
        let mut out = Vec::new();
        let mut input = &raw[..];
        if raw.starts_with(&[0xFD, b'7', b'z', b'X', b'Z', 0x00]) {
            lzma_rs::xz_decompress(&mut input, &mut out).context("xz")?;
        } else {
            lzma_rs::lzma_decompress(&mut input, &mut out).context("lzma")?;
        }
        Ok(out)
    }

    /// The 4-bit form of a tensor, or `None` when it is stored another way.
    ///
    /// The first call decodes the tensor; later calls return the cached copy, so
    /// a loader that asks twice (GPU layout + CPU dequant) pays once.
    pub fn read_lowbit(&self, name: &str) -> Result<Option<std::sync::Arc<LowBitTensor>>> {
        if let Some(hit) = self.cache.lock().unwrap().get(name) {
            return Ok(Some(hit.clone()));
        }
        let e = self.entry(name)?;
        if e.kind != Kind::Levels2 {
            return Ok(None);
        }
        let decoded =
            std::sync::Arc::new(self.decode_levels2(name, e).with_context(|| {
                format!("{name}: decoding levels2 entry at offset {}", e.offset)
            })?);
        self.cache
            .lock()
            .unwrap()
            .insert(name.to_string(), decoded.clone());
        Ok(Some(decoded))
    }

    fn decode_levels2(&self, name: &str, e: &Entry) -> Result<LowBitTensor> {
        let (rows, cols) = e.rows_cols();
        let k = e.level_count as usize;
        ensure!((1..=3).contains(&k), "{name}: unsupported level count {k}");
        let levels_bytes = self.field(e, 0, e.lev, e.zipped_levels)?;
        ensure!(
            levels_bytes.len() == rows * k * 2,
            "{name}: level table is {} bytes, expected {}",
            levels_bytes.len(),
            rows * k * 2
        );
        let meta = self.field(e, e.lev, e.meta, false)?;
        let words = self.field(e, e.lev + e.meta, e.words, false)?;

        let mut levels = vec![0u16; rows * 4];
        for r in 0..rows {
            for i in 0..k {
                let b = &levels_bytes[2 * (r * k + i)..2 * (r * k + i) + 2];
                levels[r * 4 + i] = u16::from_le_bytes([b[0], b[1]]);
            }
        }

        // Column classes: four bits per column, high nibble for even columns.
        let class_bytes = (cols * 4).div_ceil(8);
        ensure!(meta.len() >= class_bytes, "{name}: class table is short");
        let class = |col: usize| -> usize {
            let byte = meta[col / 2];
            (if col % 2 == 0 { byte >> 4 } else { byte & 0xF }) as usize
        };
        let symbols = 2 * k + 1;
        let counts: Vec<u32> = meta[class_bytes..]
            .chunks_exact(4)
            .take(16 * symbols)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ensure!(
            counts.len() == 16 * symbols,
            "{name}: symbol tables are short ({} of {})",
            counts.len(),
            16 * symbols
        );
        let word_stream: Vec<u32> = words
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut codes = vec![0u8; rows * cols];
        let mut decoder = DefaultAnsCoder::from_compressed(word_stream)
            .map_err(|_| anyhow::anyhow!("{name}: truncated ANS stream"))?;
        for c in 0..16 {
            let columns: Vec<usize> = (0..cols).filter(|&i| class(i) == c).collect();
            if columns.is_empty() {
                continue;
            }
            let table = &counts[c * symbols..(c + 1) * symbols];
            let total: f64 = table.iter().map(|&v| v as f64).sum();
            ensure!(total > 0.0, "{name}: empty symbol table for class {c}");
            let probs: Vec<f64> = table.iter().map(|&v| v as f64 / total).collect();
            let model =
                DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                    &probs, None,
                )
                .map_err(|_| anyhow::anyhow!("{name}: invalid symbol table"))?;
            let n = rows * columns.len();
            let decoded: Vec<usize> = decoder
                .decode_iid_symbols(n, &model)
                .collect::<Result<_, _>>()
                .map_err(|_| anyhow::anyhow!("{name}: ANS decode failed"))?;
            for (j, &col) in columns.iter().enumerate() {
                for r in 0..rows {
                    codes[r * cols + col] = decoded[j * rows + r] as u8;
                }
            }
        }

        let codes_ld = cols.div_ceil(8);
        let mut packed = vec![0u32; rows * codes_ld];
        for r in 0..rows {
            for col in 0..cols {
                packed[r * codes_ld + col / 8] |= (codes[r * cols + col] as u32) << (4 * (col % 8));
            }
        }
        Ok(LowBitTensor {
            codes: packed,
            levels,
            k: k as u32,
            rows,
            cols,
        })
    }

    /// The tensor as floating-point values: the 4-bit layers are expanded to the
    /// exact fp16 levels they index, the integer layers to `q * row_scale`.
    pub fn read(&self, name: &str) -> Result<RawTensor> {
        self.read_inner(name)
            .with_context(|| format!("{name}: reading container entry"))
    }

    fn read_inner(&self, name: &str) -> Result<RawTensor> {
        let e = self.entry(name)?;
        let shape = e.shape.clone();
        match e.kind {
            Kind::Levels2 => {
                let lb = self.read_lowbit(name)?.context("levels2 tensor")?;
                let codes_ld = lb.cols.div_ceil(8);
                let mut values = vec![0f32; lb.rows * lb.cols];
                for r in 0..lb.rows {
                    for col in 0..lb.cols {
                        let v = lb.codes[r * codes_ld + col / 8];
                        let code = ((v >> (4 * (col % 8))) & 0xF) as usize;
                        let k = lb.k as usize;
                        values[r * lb.cols + col] = expand(code, k, &lb.levels[r * 4..r * 4 + k]);
                    }
                }
                Ok(f32_tensor(values, shape))
            }
            Kind::AnsInt { bits } => {
                let scale_bytes = self.field(e, 0, e.scale, e.zipped_scale)?;
                let scales: Vec<f32> = scale_bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect();
                let meta = self.field(e, e.scale, e.meta, false)?;
                let counts: Vec<f64> = meta
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect();
                let total: f64 = counts.iter().sum();
                ensure!(total > 0.0, "{name}: empty symbol table");
                let probs: Vec<f64> = counts.iter().map(|&v| v / total).collect();
                let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                    &probs, None,
                )
                .map_err(|_| anyhow::anyhow!("{name}: invalid symbol table"))?;
                let words = self.field(e, e.scale + e.meta, e.words, false)?;
                let stream: Vec<u32> = words
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let n = e.n();
                let mut decoder = DefaultAnsCoder::from_compressed(stream)
                    .map_err(|_| anyhow::anyhow!("{name}: truncated ANS stream"))?;
                let symbols: Vec<usize> = decoder
                    .decode_iid_symbols(n, &model)
                    .collect::<Result<_, _>>()
                    .map_err(|_| anyhow::anyhow!("{name}: ANS decode failed"))?;
                let bias = (1i64 << (bits - 1)) - 1;
                let rows = scales.len();
                let per_row = n / rows.max(1);
                let mut values = vec![0f32; n];
                for i in 0..n {
                    let q = (symbols[i] as i64 - bias) as f32;
                    values[i] = q * scales[i / per_row.max(1)];
                }
                Ok(f32_tensor(values, shape))
            }
            Kind::Raw => {
                let bytes = self.field(e, 0, e.words, e.zipped_raw)?;
                let width = match e.raw_dtype {
                    safetensors::Dtype::I64 => 8,
                    safetensors::Dtype::I32 => 4,
                    _ => 2,
                };
                ensure!(
                    bytes.len() == e.n() * width,
                    "{name}: {} bytes for {} values of {width} bytes",
                    bytes.len(),
                    e.n()
                );
                Ok(RawTensor {
                    data: Bytes::from(bytes),
                    shape,
                    dtype: e.raw_dtype,
                })
            }
        }
    }

    /// Decodes every 4-bit tensor, in parallel on native targets.
    ///
    /// ANS streams are sequential within a tensor but tensors are independent, so
    /// this is where the load-time win is: one tensor per thread.
    pub fn prefetch(&self) -> Result<()> {
        let names: Vec<&String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.kind == Kind::Levels2)
            .map(|(n, _)| n)
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            names
                .par_iter()
                .try_for_each(|n| self.read_lowbit(n).map(|_| ()))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for n in names {
                self.read_lowbit(n)?;
            }
        }
        Ok(())
    }
}

/// Where the container bytes live.
enum Source {
    File(File),
    /// Fetched by the caller (the wasm build): the whole `model.bark` in memory.
    Memory(std::sync::Arc<Vec<u8>>),
}

/// Reads and validates the container header, returning its raw JSON bytes.
fn read_header(reader: &mut (impl Read + Seek)) -> Result<Vec<u8>> {
    let mut len = [0u8; 8];
    reader.seek(std::io::SeekFrom::Start(0))?;
    reader.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    ensure!(
        header_len > 0 && header_len < (64 << 20),
        "header claims {header_len} bytes"
    );
    let mut header = vec![0u8; header_len as usize];
    reader.read_exact(&mut header)?;
    let head: serde_json::Value = serde_json::from_slice(&header).context("header is not JSON")?;
    let format = head["format"].as_str().unwrap_or("");
    ensure!(
        format == "bark-levels-v2" || format == "bark-levels-v3",
        "unsupported container format {format:?}"
    );
    Ok(header)
}

/// Turns the JSON index into per-tensor entries, starting at `base` bytes.
fn parse_index(header: &[u8], base: u64) -> Result<HashMap<String, Entry>> {
    let head: serde_json::Value = serde_json::from_slice(header)?;
    let mut entries = HashMap::new();
    let mut offset = base;
    for e in head["index"].as_array().context("container index")? {
        let name = e["n"].as_str().context("tensor name")?.to_string();
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .context("tensor shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let kind = match e["k"].as_str().unwrap_or("") {
            "levels2" => Kind::Levels2,
            k if k.starts_with("ansint") => Kind::AnsInt {
                bits: k[6..].parse().with_context(|| format!("{name}: {k}"))?,
            },
            "raw" => Kind::Raw,
            other => bail!("{name}: unsupported entry kind {other:?}"),
        };
        let field = |key: &str| e[key].as_u64().unwrap_or(0);
        let flag = |key: &str| {
            e.get(key)
                .map(|v| v.as_bool().unwrap_or_else(|| v.as_u64().unwrap_or(0) != 0))
                .unwrap_or(false)
        };
        let entry = Entry {
            kind,
            level_count: field("K") as u32,
            shape,
            offset,
            lev: field("lev"),
            meta: field("meta"),
            words: field("b"),
            scale: field("s"),
            zipped_levels: flag("zl"),
            zipped_scale: flag("zs"),
            zipped_raw: flag("zr"),
            raw_dtype: match e["dtype"].as_str().unwrap_or("") {
                "torch.int64" => safetensors::Dtype::I64,
                "torch.int32" => safetensors::Dtype::I32,
                _ => safetensors::Dtype::F16,
            },
        };
        offset += entry.lev + entry.meta + entry.words + entry.scale;
        entries.insert(name, entry);
    }
    Ok(entries)
}

/// `x.pipe(f)` — keeps the read path a one-liner in both branches.
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

/// Reads `buf.len()` bytes at `offset` without moving the file cursor.
#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset).and_then(|n| {
        if n == buf.len() {
            Ok(())
        } else {
            Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))
        }
    })
}

#[cfg(not(any(unix, windows)))]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// `+level[code-1]` for `1..=k`, `-level[code-k-1]` above that, zero for code 0.
fn expand(code: usize, k: usize, levels: &[u16]) -> f32 {
    if code == 0 {
        return 0.0;
    }
    let (mag, negative) = if code <= k {
        (code - 1, false)
    } else {
        (code - k - 1, true)
    };
    let value = levels
        .get(mag)
        .map_or(0.0, |&bits| half::f16::from_bits(bits).to_f32());
    if negative { -value } else { value }
}

fn f32_tensor(values: Vec<f32>, shape: Vec<usize>) -> RawTensor {
    RawTensor {
        data: Bytes::from(bytemuck::cast_slice(&values).to_vec()),
        shape,
        dtype: safetensors::Dtype::F32,
    }
}
