//! Splitting work across cores — except in the browser.
//!
//! The CPU backend is tuned for a many-core desktop: every "if it is big enough,
//! hand it to rayon" branch exists next to a serial one. `wasm32-unknown-unknown`
//! has no threads at all (`std::thread` aborts), so the browser build keeps the
//! serial branches and runs one core.
//!
//! `SPLIT` is that switch: native code is unchanged (it is `true`), wasm reads
//! `false` and every `if SPLIT && big_enough` falls through to its serial path.

/// Whether the CPU backend may use rayon in this build.
#[cfg(not(target_arch = "wasm32"))]
pub const SPLIT: bool = true;

/// Whether the CPU backend may use rayon in this build.
#[cfg(target_arch = "wasm32")]
pub const SPLIT: bool = false;

/// Worker count for the block-splitting heuristics (1 in the browser).
#[cfg(not(target_arch = "wasm32"))]
pub fn num_threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// Worker count for the block-splitting heuristics (1 in the browser).
#[cfg(target_arch = "wasm32")]
pub fn num_threads() -> usize {
    1
}
