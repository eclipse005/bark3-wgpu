//! Decoded-weight cache in the browser's origin-private file system.
//!
//! A compressed release spends its load time decoding, not uploading: the
//! container's ANS/LZMA payloads cost about ten seconds for the 180 MB Plus
//! checkpoint, while the GPU upload of the result is a fraction of that. The
//! decoded tensors are deterministic and only change with the release, so they
//! are worth keeping: a second load is a file read plus the upload.
//!
//! One flat file per checkpoint, keyed by the fingerprint of the model bytes.
//! A wrong key costs a re-decode, never a wrong model.
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::bark::LowBitTensor;
use crate::raw::RawTensor;
use crate::weights::WeightSource;
use crate::weights_io::{Layout, ModelInfo};

const MAGIC: &[u8] = b"B3WCACH";
const END_MAGIC: &[u8] = b"B3WEND01";
const VERSION: u32 = 1;

/// Fingerprint of a checkpoint: length, release name and three 64 KiB samples.
///
/// Hashing all 180 MB would cost a good part of the decode it is meant to save.
#[must_use]
pub fn fingerprint(data: &[u8], release: Option<&str>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&(data.len() as u64).to_le_bytes());
    if let Some(release) = release {
        eat(release.as_bytes());
    }
    const SAMPLE: usize = 64 * 1024;
    for start in [0usize, data.len() / 2, data.len().saturating_sub(SAMPLE)] {
        let end = (start + SAMPLE).min(data.len());
        eat(&data[start..end]);
    }
    hash
}

/// One decoded tensor, in the form the loader asked for.
enum Payload {
    /// Returned by WeightSource::read.
    Raw(RawTensor),
    /// Returned by WeightSource::read_lowbit.
    LowBit(Arc<LowBitTensor>),
    /// The source does not hold this tensor in low-bit form.
    Absent,
}

/// Serves tensors straight out of a written cache file.
pub struct CachedSource {
    map: HashMap<String, Payload>,
    info: ModelInfo,
}

impl WeightSource for CachedSource {
    fn read(&self, name: &str) -> Result<RawTensor> {
        match self.map.get(name) {
            Some(Payload::Raw(t)) => Ok(t.clone()),
            Some(_) => bail!("{name} was cached as low-bit but is read as float"),
            None => bail!("{name} is missing from the weight cache"),
        }
    }

    fn read_lowbit(&self, name: &str) -> Result<Option<Arc<LowBitTensor>>> {
        match self.map.get(name) {
            Some(Payload::LowBit(t)) => Ok(Some(t.clone())),
            _ => Ok(None),
        }
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }
}

/// Records what a real source decodes, so the result can be written back.
pub struct Recorder {
    entries: Rc<RefCell<Vec<(String, Payload)>>>,
    info: ModelInfo,
}

struct Recording {
    inner: Box<dyn WeightSource>,
    entries: Rc<RefCell<Vec<(String, Payload)>>>,
}

impl WeightSource for Recording {
    fn read(&self, name: &str) -> Result<RawTensor> {
        let tensor = self.inner.read(name)?;
        self.entries
            .borrow_mut()
            .push((name.to_string(), Payload::Raw(tensor.clone())));
        Ok(tensor)
    }

    fn read_lowbit(&self, name: &str) -> Result<Option<Arc<LowBitTensor>>> {
        let tensor = self.inner.read_lowbit(name)?;
        let entry = match &tensor {
            Some(t) => Payload::LowBit(t.clone()),
            None => Payload::Absent,
        };
        self.entries.borrow_mut().push((name.to_string(), entry));
        Ok(tensor)
    }

    fn prefetch(&self) -> Result<()> {
        self.inner.prefetch()
    }

    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }
}

/// Wraps a source so its decoded tensors can be persisted after the load.
pub fn record(source: Box<dyn WeightSource>) -> (Box<dyn WeightSource>, Recorder) {
    let info = source.info().clone();
    let entries = Rc::new(RefCell::new(Vec::new()));
    let source = Box::new(Recording {
        entries: entries.clone(),
        inner: source,
    });
    (source, Recorder { entries, info })
}

impl Recorder {
    /// Writes everything this source decoded to the cache file for the key.
    pub async fn persist(&self, key: u64) -> Result<()> {
        let dir = directory().await?;
        let mut options = web_sys::FileSystemGetFileOptions::new();
        options.set_create(true);
        let handle = JsFuture::from(dir.get_file_handle_with_options(&file_name(key), &options))
            .await
            .map_err(|e| anyhow!("creating the cache file: {e:?}"))?
            .dyn_into::<web_sys::FileSystemFileHandle>()
            .map_err(|e| anyhow!("file handle: {e:?}"))?;
        let stream = JsFuture::from(handle.create_writable())
            .await
            .map_err(|e| anyhow!("create_writable: {e:?}"))?
            .dyn_into::<web_sys::FileSystemWritableFileStream>()
            .map_err(|e| anyhow!("writable stream: {e:?}"))?;

        let entries = self.entries.borrow();
        let total: usize = entries
            .iter()
            .map(|(name, p)| match p {
                Payload::Raw(t) => name.len() + t.data.len(),
                Payload::LowBit(t) => name.len() + t.codes.len() * 4 + t.levels.len() * 2,
                Payload::Absent => name.len(),
            })
            .sum();
        crate::browser::log(&format!(
            "[wasm] cache write {} entries, {:.1} MB",
            entries.len(),
            total as f64 / 1e6
        ));
        let mut header = Vec::with_capacity(MAGIC.len() + 8);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        write(&stream, &header).await?;

        for (name, payload) in entries.iter() {
            let mut meta = Vec::with_capacity(16 + name.len());
            meta.extend_from_slice(&(name.len() as u16).to_le_bytes());
            meta.extend_from_slice(name.as_bytes());
            match payload {
                Payload::Absent => meta.push(2),
                Payload::Raw(t) => {
                    meta.push(0);
                    meta.push(dtype_tag(t.dtype));
                    meta.push(t.shape.len() as u8);
                    for dim in &t.shape {
                        meta.extend_from_slice(&(*dim as u32).to_le_bytes());
                    }
                    meta.extend_from_slice(&(t.data.len() as u32).to_le_bytes());
                }
                Payload::LowBit(t) => {
                    meta.push(1);
                    meta.extend_from_slice(&t.k.to_le_bytes());
                    meta.extend_from_slice(&(t.rows as u32).to_le_bytes());
                    meta.extend_from_slice(&(t.cols as u32).to_le_bytes());
                    meta.extend_from_slice(&((t.codes.len() * 4) as u32).to_le_bytes());
                    meta.extend_from_slice(&((t.levels.len() * 2) as u32).to_le_bytes());
                }
            }
            write(&stream, &meta).await?;
            match payload {
                Payload::Raw(t) => write(&stream, &t.data).await?,
                Payload::LowBit(t) => {
                    write(&stream, bytemuck::cast_slice(&t.codes)).await?;
                    write(&stream, bytemuck::cast_slice(&t.levels)).await?;
                }
                Payload::Absent => {}
            }
        }

        let mut tail = info_bytes(&self.info);
        tail.extend_from_slice(END_MAGIC);
        write(&stream, &tail).await?;
        JsFuture::from(stream.unchecked_ref::<web_sys::WritableStream>().close())
            .await
            .map_err(|e| anyhow!("closing the cache file: {e:?}"))?;
        Ok(())
    }
}

fn info_bytes(info: &ModelInfo) -> Vec<u8> {
    let mut out = Vec::new();
    let release = info.release.as_deref().unwrap_or("");
    out.extend_from_slice(&(release.len() as u16).to_le_bytes());
    out.extend_from_slice(release.as_bytes());
    out.push(match info.layout {
        Layout::Bf16 => 0,
        Layout::F16 => 1,
        Layout::F32 => 2,
        Layout::LowBit => 3,
    });
    out.extend_from_slice(&(info.tensors as u32).to_le_bytes());
    out.extend_from_slice(&info.weights.to_le_bytes());
    out.extend_from_slice(&(info.layers as u32).to_le_bytes());
    out.extend_from_slice(&(info.hidden as u32).to_le_bytes());
    out.extend_from_slice(&(info.ffn as u32).to_le_bytes());
    out.extend_from_slice(&(info.vocab as u32).to_le_bytes());
    out
}

fn dtype_tag(dtype: safetensors::Dtype) -> u8 {
    match dtype {
        safetensors::Dtype::BF16 => 1,
        safetensors::Dtype::F32 => 2,
        safetensors::Dtype::U32 => 3,
        _ => 0,
    }
}

fn dtype_from_tag(tag: u8) -> safetensors::Dtype {
    match tag {
        1 => safetensors::Dtype::BF16,
        2 => safetensors::Dtype::F32,
        3 => safetensors::Dtype::U32,
        _ => safetensors::Dtype::F16,
    }
}

/// Reads the cache file for the key, or None when it is absent or stale.
pub async fn load(key: u64) -> Result<Option<Box<dyn WeightSource>>> {
    let dir = directory().await?;
    let handle = match JsFuture::from(dir.get_file_handle(&file_name(key))).await {
        Ok(handle) => handle,
        Err(_) => return Ok(None),
    };
    let file = JsFuture::from(
        handle
            .dyn_into::<web_sys::FileSystemFileHandle>()
            .map_err(|e| anyhow!("file handle: {e:?}"))?
            .get_file(),
    )
    .await
    .map_err(|e| anyhow!("reading the cache file: {e:?}"))?
    .dyn_into::<web_sys::File>()
    .map_err(|e| anyhow!("file: {e:?}"))?;
    let started = crate::clock::now();
    let size = file.size() as usize;
    let buffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|e| anyhow!("cache bytes: {e:?}"))?
        .dyn_into::<js_sys::ArrayBuffer>()
        .map_err(|e| anyhow!("array buffer: {e:?}"))?;
    let read_s = started.elapsed().as_secs_f32();
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
    let copied_s = started.elapsed().as_secs_f32();
    crate::browser::log(&format!(
        "[wasm] cache read {:.1} MB: read {read_s:.2}s copy {:.2}s",
        size as f64 / 1e6,
        copied_s - read_s
    ));
    let parse_at = crate::clock::now();
    let parsed = parse(&bytes);
    crate::browser::log(&format!(
        "[wasm] cache parse {:.2}s",
        parse_at.elapsed().as_secs_f32()
    ));
    match parsed {
        Ok(Some(source)) => Ok(Some(Box::new(source))),
        Ok(None) => Ok(None),
        Err(e) => {
            crate::browser::log(&format!(
                "[wasm] weight cache unusable ({e:#}); re-decoding"
            ));
            Ok(None)
        }
    }
}

fn parse(bytes: &[u8]) -> Result<Option<CachedSource>> {
    if bytes.len() < MAGIC.len() + 8 || &bytes[..MAGIC.len()] != MAGIC {
        return Ok(None);
    }
    let mut pos = MAGIC.len();
    if read_u32(bytes, &mut pos)? != VERSION {
        return Ok(None);
    }
    let count = read_u32(bytes, &mut pos)? as usize;
    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u16(bytes, &mut pos)? as usize;
        let name = std::str::from_utf8(slice(bytes, &mut pos, name_len)?)?.to_string();
        match read_u8(bytes, &mut pos)? {
            2 => {
                map.insert(name, Payload::Absent);
            }
            0 => {
                let dtype = dtype_from_tag(read_u8(bytes, &mut pos)?);
                let ndim = read_u8(bytes, &mut pos)? as usize;
                let mut shape = Vec::with_capacity(ndim);
                for _ in 0..ndim {
                    shape.push(read_u32(bytes, &mut pos)? as usize);
                }
                let len = read_u32(bytes, &mut pos)? as usize;
                let data = Bytes::copy_from_slice(slice(bytes, &mut pos, len)?);
                map.insert(name, Payload::Raw(RawTensor { data, shape, dtype }));
            }
            1 => {
                let k = read_u32(bytes, &mut pos)?;
                let rows = read_u32(bytes, &mut pos)? as usize;
                let cols = read_u32(bytes, &mut pos)? as usize;
                let codes_len = read_u32(bytes, &mut pos)? as usize;
                let levels_len = read_u32(bytes, &mut pos)? as usize;
                // Payload offsets are not word-aligned, so read the words
                // explicitly instead of casting the byte slice.
                let codes = slice(bytes, &mut pos, codes_len)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let levels = slice(bytes, &mut pos, levels_len)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                map.insert(
                    name,
                    Payload::LowBit(Arc::new(LowBitTensor {
                        codes,
                        levels,
                        k,
                        rows,
                        cols,
                    })),
                );
            }
            other => bail!("unknown payload kind {other}"),
        }
    }
    let release_len = read_u16(bytes, &mut pos)? as usize;
    let release = String::from_utf8(slice(bytes, &mut pos, release_len)?.to_vec())?;
    let layout = match read_u8(bytes, &mut pos)? {
        1 => Layout::F16,
        2 => Layout::F32,
        3 => Layout::LowBit,
        _ => Layout::Bf16,
    };
    let tensors = read_u32(bytes, &mut pos)? as usize;
    let weights = read_u64(bytes, &mut pos)?;
    let layers = read_u32(bytes, &mut pos)? as usize;
    let hidden = read_u32(bytes, &mut pos)? as usize;
    let ffn = read_u32(bytes, &mut pos)? as usize;
    let vocab = read_u32(bytes, &mut pos)? as usize;
    if bytes.get(pos..pos + END_MAGIC.len()) != Some(END_MAGIC) {
        bail!("truncated cache file");
    }
    Ok(Some(CachedSource {
        map,
        info: ModelInfo {
            release: if release.is_empty() {
                None
            } else {
                Some(release)
            },
            layout,
            tensors,
            weights,
            layers,
            hidden,
            ffn,
            vocab,
        },
    }))
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    Ok(slice(bytes, pos, 1)?[0])
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    let s = slice(bytes, pos, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    let s = slice(bytes, pos, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let s = slice(bytes, pos, 8)?;
    let mut out = [0u8; 8];
    out.copy_from_slice(s);
    Ok(u64::from_le_bytes(out))
}

fn slice<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = pos.checked_add(len).context("cache offset overflow")?;
    let out = bytes.get(*pos..end).context("truncated cache file")?;
    *pos = end;
    Ok(out)
}

async fn write(stream: &web_sys::FileSystemWritableFileStream, bytes: &[u8]) -> Result<()> {
    let promise = stream
        .write_with_u8_array(bytes)
        .map_err(|e| anyhow!("write: {e:?}"))?;
    JsFuture::from(promise)
        .await
        .map_err(|e| anyhow!("write: {e:?}"))?;
    Ok(())
}

fn file_name(key: u64) -> String {
    format!("bark3-weights-{key:016x}.bin")
}

/// The origin's private file system.
async fn directory() -> Result<web_sys::FileSystemDirectoryHandle> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no window"))?;
    let promise = window.navigator().storage().get_directory();
    JsFuture::from(promise)
        .await
        .map_err(|e| anyhow!("storage.getDirectory: {e:?}"))?
        .dyn_into::<web_sys::FileSystemDirectoryHandle>()
        .map_err(|e| anyhow!("directory handle: {e:?}"))
}
