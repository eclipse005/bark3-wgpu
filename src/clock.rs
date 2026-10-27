//! A monotonic clock that also exists in the browser.
//!
//! `std::time::Instant::now()` panics on `wasm32-unknown-unknown` ("time not
//! implemented on this platform"), and the pipeline timestamps every stage. This
//! thin wrapper keeps the native behaviour identical (`web-time` re-exports
//! `std::time::Instant` there) and uses `performance.now()` in the browser.

pub use web_time::Instant;

/// `Instant::now()`, named so call sites read as a clock read.
#[must_use]
pub fn now() -> Instant {
    Instant::now()
}
