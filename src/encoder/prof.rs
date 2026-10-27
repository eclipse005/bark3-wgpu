//! Opt-in GPU timestamp attribution for the encoder pass.
//!
//! Every operator records one timestamp right before it is dispatched, so the
//! interval that follows a mark is that operator's execution time. Marks are
//! collected in a single device query set, resolved once at the end of the pass
//! and aggregated per operator label.
//!
//! `BARK3_ENC_PROFILE=1` enables it. Production runs never allocate the query
//! set, so the timings cost nothing when the variable is unset.

use std::cell::Cell;
use std::cell::RefCell;

use anyhow::Result;

use crate::gpu::Gpu;

/// Marks per pass. wgpu caps a query set at 4096 timestamps, which is ample:
/// a 180 s utterance issues roughly 1.2k dispatches.
const CAPACITY: u32 = 4096;

pub struct Prof {
    queries: wgpu::QuerySet,
    ticks: wgpu::Buffer,
    labels: RefCell<Vec<&'static str>>,
    cursor: Cell<u32>,
    period: f32,
}

impl Prof {
    /// Returns `None` when the device cannot timestamp inside a compute pass.
    pub fn new(gpu: &Gpu, label: &str) -> Option<Self> {
        let needed =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        if !gpu.features.contains(needed) {
            eprintln!("[enc-profile] timestamps unavailable, profiling disabled");
            return None;
        }
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(label),
            ty: wgpu::QueryType::Timestamp,
            count: CAPACITY,
        });
        let ticks = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_profile_ticks"),
            size: CAPACITY as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            queries,
            ticks,
            labels: RefCell::new(Vec::with_capacity(1024)),
            cursor: Cell::new(0),
            period: gpu.queue.get_timestamp_period(),
        })
    }

    pub fn mark(&self, cp: &mut wgpu::ComputePass<'_>, label: &'static str) {
        let index = self.cursor.get();
        if index >= CAPACITY {
            return;
        }
        cp.write_timestamp(&self.queries, index);
        self.labels.borrow_mut().push(label);
        self.cursor.set(index + 1);
    }

    /// Resolve the collected marks. Safe to call from a trailing encoder: every
    /// earlier submission has been queued, so their timestamps are readable.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.cursor.get();
        if count > 1 {
            encoder.resolve_query_set(&self.queries, 0..count, &self.ticks, 0);
        }
    }

    /// Aggregate the intervals and print them heaviest first.
    pub fn report(&self, gpu: &Gpu) -> Result<()> {
        let count = self.cursor.get();
        if count < 2 {
            return Ok(());
        }
        let bytes = gpu.readback(&self.ticks, count as u64 * 8)?;
        let labels = self.labels.borrow();
        let mut totals: Vec<(&'static str, f64, u32)> = Vec::new();
        // `get_timestamp_period` is nanoseconds per tick.
        let to_ms = self.period as f64 / 1e6;
        for index in 0..(count - 1) as usize {
            let start = read_u64(&bytes, index * 8) as f64;
            let end = read_u64(&bytes, (index + 1) * 8) as f64;
            let millis = (end - start) * to_ms;
            let label = labels[index];
            match totals.iter_mut().find(|(name, _, _)| *name == label) {
                Some(entry) => {
                    entry.1 += millis;
                    entry.2 += 1;
                }
                None => totals.push((label, millis, 1)),
            }
        }
        totals.sort_by(|a, b| b.1.total_cmp(&a.1));
        let span: f64 = totals.iter().map(|(_, millis, _)| millis).sum();
        println!("[enc-profile] {span:.1}ms over {} dispatches", count - 1);
        for (label, millis, calls) in &totals {
            println!(
                "[enc-profile]   {label:<16} {millis:>8.1}ms  {calls:>5}x  {:>5.1}%  {:>8.1}us/call",
                millis / span * 100.0,
                millis / *calls as f64 * 1000.0
            );
        }
        Ok(())
    }

    pub fn reset(&self) {
        self.cursor.set(0);
        self.labels.borrow_mut().clear();
    }
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(word)
}
