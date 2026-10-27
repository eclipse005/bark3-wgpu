//! Scoped worker pool for TDT decode.
//!
//! The safe wrappers in this module validate every slice length before crossing
//! the raw-pointer boundary. `submit` blocks until all workers finish, so the
//! borrowed buffers remain alive and are never accessed by two jobs at once.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

use super::{DH, GS, TdtDecoder, dot16, sigmoid};
use crate::cfg;

const PHASE_GATES: usize = 0;
const PHASE_UPDATE: usize = 1;
const PHASE_JOINT: usize = 2;
const PHASE_PROJECT: usize = 3;
const MAX_WORKERS: usize = 32;

type JointPartial = (f32, u32, f32, u32);

/// Spin-pool used by the TDT decoder. Workers keep a fixed row range, so each
/// worker's LSTM weight slice stays resident in its own L2.
pub(super) struct LstmPool<'a> {
    dec: &'a TdtDecoder,
    n: usize,
    epoch: AtomicUsize,
    done: AtomicUsize,
    stop_flag: AtomicBool,
    phase: AtomicUsize,
    layer: AtomicUsize,
    p0: AtomicPtr<f32>,
    p1: AtomicPtr<f32>,
    p2: AtomicPtr<f32>,
    parts: UnsafeCell<[JointPartial; MAX_WORKERS]>,
}

// SAFETY: `LstmPool` is crate-private and used with one producer thread inside
// `TdtDecoder::decode`. Each worker writes only its own row range and partial;
// SeqCst job/completion handoffs protect all buffer accesses between jobs.
unsafe impl Send for LstmPool<'_> {}
// SAFETY: See the synchronization and disjoint-partition contract above.
unsafe impl Sync for LstmPool<'_> {}

impl<'a> LstmPool<'a> {
    pub(super) fn new(dec: &'a TdtDecoder, n: usize) -> Self {
        assert!((1..=MAX_WORKERS).contains(&n));
        Self {
            dec,
            n,
            epoch: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop_flag: AtomicBool::new(false),
            phase: AtomicUsize::new(0),
            layer: AtomicUsize::new(0),
            p0: AtomicPtr::new(std::ptr::null_mut()),
            p1: AtomicPtr::new(std::ptr::null_mut()),
            p2: AtomicPtr::new(std::ptr::null_mut()),
            parts: UnsafeCell::new([(f32::NEG_INFINITY, 0, f32::NEG_INFINITY, 0); MAX_WORKERS]),
        }
    }

    pub(super) fn gates(&self, layer: usize, xin: &[f32], hin: &[f32], out: &mut [f32]) {
        assert!(layer < 2);
        assert_eq!(xin.len(), DH);
        assert_eq!(hin.len(), DH);
        assert_eq!(out.len(), GS);
        self.submit(
            PHASE_GATES,
            layer,
            xin.as_ptr() as *mut f32,
            hin.as_ptr() as *mut f32,
            out.as_mut_ptr(),
        );
    }

    pub(super) fn update(&self, gates: &[f32], h: &mut [f32], c: &mut [f32]) {
        assert_eq!(gates.len(), GS);
        assert_eq!(h.len(), DH);
        assert_eq!(c.len(), DH);
        self.submit(
            PHASE_UPDATE,
            0,
            gates.as_ptr() as *mut f32,
            h.as_mut_ptr(),
            c.as_mut_ptr(),
        );
    }

    pub(super) fn project(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), DH);
        assert_eq!(out.len(), DH);
        self.submit(
            PHASE_PROJECT,
            0,
            x.as_ptr() as *mut f32,
            out.as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }

    pub(super) fn joint(&self, x: &[f32]) -> (u32, u32) {
        assert_eq!(x.len(), DH);
        self.submit(
            PHASE_JOINT,
            0,
            std::ptr::null_mut(),
            x.as_ptr() as *mut f32,
            std::ptr::null_mut(),
        );
        // SAFETY: all workers finished in `submit`; each initialized only its
        // own slot, and no worker writes again until the next submission.
        let parts = unsafe { &*self.parts.get() };
        let mut best_token = (f32::NEG_INFINITY, 0u32);
        let mut best_duration = (f32::NEG_INFINITY, 0u32);
        for &(token_score, token_id, duration_score, duration_id) in &parts[..self.n] {
            if token_score > best_token.0 {
                best_token = (token_score, token_id);
            }
            if duration_score > best_duration.0 {
                best_duration = (duration_score, duration_id);
            }
        }
        (best_token.1, best_duration.1 - cfg::VOCAB as u32)
    }

    fn submit(&self, phase: usize, layer: usize, p0: *mut f32, p1: *mut f32, p2: *mut f32) {
        self.phase.store(phase, Ordering::SeqCst);
        self.layer.store(layer, Ordering::SeqCst);
        self.p0.store(p0, Ordering::SeqCst);
        self.p1.store(p1, Ordering::SeqCst);
        self.p2.store(p2, Ordering::SeqCst);
        self.done.store(0, Ordering::SeqCst);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        let mut spins = 0u32;
        while self.done.load(Ordering::SeqCst) < self.n {
            if spins < 2000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    pub(super) fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn worker(&self, worker_index: usize) {
        let simd = TdtDecoder::simd_ok();
        let mut generation = 0usize;
        loop {
            while self.epoch.load(Ordering::SeqCst) == generation {
                if self.stop_flag.load(Ordering::Relaxed) {
                    return;
                }
                let mut spins = 0u32;
                while self.epoch.load(Ordering::SeqCst) == generation {
                    if self.stop_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    if spins < 4000 {
                        spins += 1;
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
            generation = self.epoch.load(Ordering::SeqCst);
            if self.stop_flag.load(Ordering::Relaxed) {
                return;
            }

            let phase = self.phase.load(Ordering::SeqCst);
            let layer = self.layer.load(Ordering::SeqCst);
            let p0 = self.p0.load(Ordering::SeqCst);
            let p1 = self.p1.load(Ordering::SeqCst);
            let p2 = self.p2.load(Ordering::SeqCst);
            match phase {
                PHASE_GATES => {
                    // SAFETY: `gates` validated all lengths and keeps the slices
                    // alive until this job completes. Workers use disjoint rows.
                    let (xin, hin, out) = unsafe {
                        (
                            std::slice::from_raw_parts(p0, DH),
                            std::slice::from_raw_parts(p1, DH),
                            std::slice::from_raw_parts_mut(p2, GS),
                        )
                    };
                    let (start, end) = chunk(worker_index, self.n, GS);
                    let wih = &self.dec.w_ih[layer * GS * DH..(layer + 1) * GS * DH];
                    let whh = &self.dec.w_hh[layer * GS * DH..(layer + 1) * GS * DH];
                    for row in start..end {
                        let wi = &wih[row * DH..(row + 1) * DH];
                        let wh = &whh[row * DH..(row + 1) * DH];
                        out[row] = self.dec.b_ih[layer * GS + row]
                            + self.dec.b_hh[layer * GS + row]
                            + dot16(wi, xin, simd)
                            + dot16(wh, hin, simd);
                    }
                }
                PHASE_UPDATE => {
                    // SAFETY: `update` validated all lengths; h/c are distinct
                    // mutable slices and workers update disjoint channels.
                    let (gates, h, c) = unsafe {
                        (
                            std::slice::from_raw_parts(p0, GS),
                            std::slice::from_raw_parts_mut(p1, DH),
                            std::slice::from_raw_parts_mut(p2, DH),
                        )
                    };
                    let (start, end) = chunk(worker_index, self.n, DH);
                    for channel in start..end {
                        let input = sigmoid(gates[channel]);
                        let forget = sigmoid(gates[DH + channel]);
                        let candidate = gates[2 * DH + channel].tanh();
                        let output = sigmoid(gates[3 * DH + channel]);
                        let cell = forget * c[channel] + input * candidate;
                        c[channel] = cell;
                        h[channel] = output * cell.tanh();
                    }
                }
                PHASE_JOINT => {
                    // SAFETY: `joint` validates x and keeps it alive until all
                    // workers finish. Each worker writes only its own partial.
                    let x = unsafe { std::slice::from_raw_parts(p1, DH) };
                    let (start, end) = chunk(worker_index, self.n, cfg::VOCAB + cfg::N_DUR);
                    let mut best_token = (f32::NEG_INFINITY, 0u32);
                    let mut best_duration = (f32::NEG_INFINITY, 0u32);
                    for row_index in start..end {
                        let row = &self.dec.head_w[row_index * DH..(row_index + 1) * DH];
                        let score = self.dec.head_b[row_index] + dot16(row, x, simd);
                        if row_index < cfg::VOCAB {
                            if score > best_token.0 {
                                best_token = (score, row_index as u32);
                            }
                        } else if score > best_duration.0 {
                            best_duration = (score, row_index as u32);
                        }
                    }
                    unsafe {
                        (*self.parts.get())[worker_index] =
                            (best_token.0, best_token.1, best_duration.0, best_duration.1);
                    }
                }
                PHASE_PROJECT => {
                    // SAFETY: `project` validates x/out; workers write disjoint
                    // output rows and only read x.
                    let (x, out) = unsafe {
                        (
                            std::slice::from_raw_parts(p0, DH),
                            std::slice::from_raw_parts_mut(p1, DH),
                        )
                    };
                    let (start, end) = chunk(worker_index, self.n, DH);
                    for (offset, value) in out[start..end].iter_mut().enumerate() {
                        let row_index = start + offset;
                        let row = &self.dec.proj_w[row_index * DH..(row_index + 1) * DH];
                        *value = self.dec.proj_b[row_index] + dot16(row, x, simd);
                    }
                }
                _ => unreachable!("invalid TDT pool phase"),
            }
            self.done.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[inline]
fn chunk(index: usize, workers: usize, total: usize) -> (usize, usize) {
    let start = index * total / workers;
    let end = (index + 1) * total / workers;
    (start, end)
}
