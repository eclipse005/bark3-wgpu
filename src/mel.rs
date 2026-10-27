//! Parakeet feature extraction (CPU): preemphasis -> STFT -> power -> 128-bin
//! log-mel (slaney) -> per-utterance normalize. Matches
//! transformers' ParakeetFeatureExtractor exactly (golden-checked).

use anyhow::Result;
use rayon::prelude::*;

use crate::cfg;

pub struct MelFrontend {
    /// (128, 257) librosa slaney filterbank, f32.
    pub mel_filters: Vec<f32>,
    /// symmetric hann window of win_length, zero-padded to n_fft (offset (512-400)/2).
    window: Vec<f32>,
}

pub struct Features {
    /// (frames, 128) log-mel, normalized, padded frames zeroed.
    pub frames: Vec<f32>,
    pub n_frames: usize, // raw frame count (1 + len/hop)
    pub n_valid: usize,  // attention-mask length (len/hop)
}

impl MelFrontend {
    pub fn new(mel_filters: Vec<f32>) -> Self {
        assert_eq!(mel_filters.len(), cfg::MEL_BINS * 257);
        let win = 400usize;
        let n_fft = 512usize;
        let mut window = vec![0f32; n_fft];
        for i in 0..win {
            // torch.hann_window(periodic=False): symmetric
            window[(n_fft - win) / 2 + i] =
                0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / (win - 1) as f32).cos());
        }
        Self {
            mel_filters,
            window,
        }
    }

    pub fn compute(&self, wave_in: &[f32]) -> Features {
        const HOP: usize = 160;
        const N_FFT: usize = 512;
        // preemphasis: x[0] stays, x[i] -= 0.97 * x[i-1]
        let mut wave = Vec::with_capacity(wave_in.len());
        for (i, &s) in wave_in.iter().enumerate() {
            wave.push(if i == 0 { s } else { s - 0.97 * wave_in[i - 1] });
        }
        let wave = &wave;
        let n_frames = 1 + wave.len() / HOP;
        let n_valid = wave.len() / HOP; // torch floor_divide semantics

        // power spectrum per frame, real FFT of size 512 (257 bins), center-padded
        let mut plans = rustfft::FftPlanner::<f32>::new();
        let fft = plans.plan_fft_forward(N_FFT);
        let spectra: Vec<Vec<f32>> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut buf = vec![rustfft::num_complex::Complex::new(0.0, 0.0); N_FFT];
                let start = t as isize * HOP as isize - N_FFT as isize / 2; // center=True, zero pad
                for (i, value) in buf.iter_mut().enumerate() {
                    let s = start + i as isize;
                    if s >= 0 && (s as usize) < wave.len() {
                        *value = rustfft::num_complex::Complex {
                            re: wave[s as usize] * self.window[i],
                            im: 0.0,
                        };
                    }
                }
                fft.process(&mut buf);
                (0..257).map(|k| buf[k].norm_sqr()).collect::<Vec<f32>>()
            })
            .collect();

        // mel (128, n_frames): filters @ power — parallel over mel rows
        let mut mel = vec![0f32; cfg::MEL_BINS * n_frames];
        let powers: Vec<&[f32]> = spectra.iter().map(|v| v.as_slice()).collect();
        mel.par_chunks_exact_mut(n_frames)
            .enumerate()
            .for_each(|(m, out)| {
                let row = &self.mel_filters[m * 257..(m + 1) * 257];
                for t in 0..n_frames {
                    let power = powers[t];
                    let mut s = 0f32;
                    for k in 0..257 {
                        if row[k] != 0.0 {
                            s += row[k] * power[k];
                        }
                    }
                    out[t] = (s + 2f32.powi(-24)).ln();
                }
            });

        // per-utterance normalize over VALID frames: sample variance (N-1)
        let mut mean = [0f32; cfg::MEL_BINS];
        let nv = n_valid.max(1);
        for m in 0..cfg::MEL_BINS {
            let mut s = 0f32;
            for t in 0..n_valid {
                s += mel[m * n_frames + t];
            }
            mean[m] = s / nv as f32;
        }
        let mut std = [0f32; cfg::MEL_BINS];
        for m in 0..cfg::MEL_BINS {
            let mut s = 0f32;
            for t in 0..n_valid {
                let d = mel[m * n_frames + t] - mean[m];
                s += d * d;
            }
            std[m] = ((s / (nv - 1).max(1) as f32) + 0.0).sqrt();
        }
        // frames-major output, padded frames zeroed
        let mut frames = vec![0f32; n_frames * cfg::MEL_BINS];
        for t in 0..n_frames {
            if t >= n_valid {
                continue;
            }
            for m in 0..cfg::MEL_BINS {
                frames[t * cfg::MEL_BINS + m] = (mel[m * n_frames + t] - mean[m]) / (std[m] + 1e-5);
            }
        }
        Features {
            frames,
            n_frames,
            n_valid,
        }
    }
}

/// Read a 16k-mono-compatible WAV via hound (any sample rate the file has is
/// returned raw; caller resamples upstream — for our fixtures everything is
/// resampled to 16k mono before hitting this, or read via ffmpeg-produced wavs).
pub fn read_wav(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    anyhow::ensure!(
        spec.sample_rate == 16000,
        "expected 16 kHz wav, got {}",
        spec.sample_rate
    );
    let ch = spec.channels as usize;
    let mut out = Vec::with_capacity(r.duration() as usize / ch);
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in r.samples::<f32>().flatten() {
                out.push(s);
            }
        }
        _ => {
            for s in r.samples::<i16>().flatten() {
                out.push(s as f32 / 32768.0);
            }
        }
    }
    if ch > 1 {
        Ok(out
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect())
    } else {
        Ok(out)
    }
}
