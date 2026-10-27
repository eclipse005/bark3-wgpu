//! Raw tensor views over mmap'ed safetensors + precision conversions.

use bytes::Bytes;

#[derive(Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
}

impl RawTensor {
    pub fn n(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read the tensor as f32 values.
    ///
    /// Checkpoints come in three precisions: `bf16` (the reference Pro
    /// release), `f16` and `f32` (decoded low-bit checkpoints, e.g. the
    /// Underdog-Bark-3.0 `.bark` container, whose weights are f32 products of a
    /// quantized level and a scale). All three widen losslessly.
    pub fn to_f32(&self) -> Vec<f32> {
        let bytes: &[u8] = &self.data;
        let n = self.n();
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            safetensors::Dtype::BF16 => {
                for i in 0..n {
                    let b = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    out.add_bf16(b);
                }
            }
            safetensors::Dtype::F16 => {
                for c in bytes.chunks_exact(2).take(n) {
                    out.push(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32());
                }
            }
            safetensors::Dtype::F32 => {
                for c in bytes.chunks_exact(4).take(n) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            safetensors::Dtype::U32 => panic!("u32 code tensors are read with to_u32"),
            other => panic!("{other:?} weights are not supported"),
        }
        out
    }

    /// 4-bit code words (low-bit checkpoints) as they are stored, one u32 per
    /// eight weights along the input dimension.
    pub fn to_u32(&self) -> Vec<u32> {
        assert_eq!(
            self.dtype,
            safetensors::Dtype::U32,
            "expected a packed u32 code tensor"
        );
        let bytes: &[u8] = &self.data;
        assert_eq!(bytes.len(), self.n() * 4, "u32 tensor has a wrong length");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Any of those precisions -> f16, packed two per u32 (low half first) for
    /// GPU storage. bf16 -> f16 is exact; f16 passes through; f32 rounds once.
    pub fn to_f16_words(&self) -> Vec<u32> {
        let bytes: &[u8] = &self.data;
        let n = self.n();
        let mut out = Vec::with_capacity(n.div_ceil(2));
        for w in 0..n.div_ceil(2) {
            let mut word = 0u32;
            for h in 0..2 {
                let i = 2 * w + h;
                if i < n {
                    let h16 = match self.dtype {
                        safetensors::Dtype::BF16 => {
                            let b = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                            f32_to_f16_bits(f32::from_bits(bf16_to_f32_bits(b)))
                        }
                        safetensors::Dtype::F16 => {
                            u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
                        }
                        safetensors::Dtype::F32 => {
                            let c = &bytes[4 * i..4 * i + 4];
                            f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        }
                        other => panic!("{other:?} weights are not supported"),
                    };
                    word |= (h16 as u32) << (16 * h);
                }
            }
            out.push(word);
        }
        out
    }

    /// f32 -> f16 words (for CPU-built tensors like mel / pos embed uploads).
    pub fn f32_to_f16_words(vals: &[f32]) -> Vec<u32> {
        let n = vals.len();
        let mut out = Vec::with_capacity(n.div_ceil(2));
        for w in 0..n.div_ceil(2) {
            let mut word = 0u32;
            for h in 0..2 {
                let i = 2 * w + h;
                if i < n {
                    word |= (f32_to_f16_bits(vals[i]) as u32) << (16 * h);
                }
            }
            out.push(word);
        }
        out
    }
}

trait PushBf16 {
    fn add_bf16(&mut self, b: u16);
}
impl PushBf16 for Vec<f32> {
    fn add_bf16(&mut self, b: u16) {
        self.push(f32::from_bits(bf16_to_f32_bits(b)));
    }
}

pub fn bf16_to_f32_bits(b: u16) -> u32 {
    ((b as u32) & 0x8000) << 16 | (((b as u32) & 0x7fff) << 16)
}

/// f32 bits -> f16 bits (round to nearest even), matching half::f16.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(dtype: safetensors::Dtype, bytes: Vec<u8>, n: usize) -> RawTensor {
        RawTensor {
            data: bytes.into(),
            shape: vec![n],
            dtype,
        }
    }

    /// The three precisions a checkpoint can arrive in must widen identically:
    /// the values below are exact in every one of them, so any difference is a
    /// conversion bug, not rounding.
    #[test]
    fn bf16_f16_f32_sources_agree() {
        let values = [0.0f32, 1.0, -2.5, 0.125, 1024.0, -0.0625];
        let n = values.len();
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f16: Vec<u8> = values
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect();
        let f32b: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let as_f32 = [
            raw(safetensors::Dtype::BF16, bf16.clone(), n).to_f32(),
            raw(safetensors::Dtype::F16, f16.clone(), n).to_f32(),
            raw(safetensors::Dtype::F32, f32b.clone(), n).to_f32(),
        ];
        for width in &as_f32[1..] {
            assert_eq!(width, &as_f32[0]);
        }

        let words = [
            raw(safetensors::Dtype::BF16, bf16, n).to_f16_words(),
            raw(safetensors::Dtype::F16, f16, n).to_f16_words(),
            raw(safetensors::Dtype::F32, f32b, n).to_f16_words(),
        ];
        assert_eq!(words[1], words[0]);
        assert_eq!(words[2], words[0]);
        // Two halves per word, low half first.
        assert_eq!(words[0][0] as u16, f32_to_f16_bits(0.0));
        assert_eq!((words[0][0] >> 16) as u16, f32_to_f16_bits(1.0));
    }
}
