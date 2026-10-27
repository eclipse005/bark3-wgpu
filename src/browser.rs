//! `console.log` in the browser, stderr everywhere else.
//!
//! Load-time progress is the only way to see where a browser stalls: a wasm
//! `eprintln!` goes nowhere, and the page has no handle on the Rust side while
//! `create()` is still running.

/// Log one line to the browser console (or stderr natively).
#[cfg(target_arch = "wasm32")]
pub fn log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

/// Log one line to the browser console (or stderr natively).
#[cfg(not(target_arch = "wasm32"))]
pub fn log(msg: &str) {
    // Load-path breadcrumbs stay out of the CLI's own output unless asked for.
    if std::env::var_os("BARK3_TRACE").is_some() {
        eprintln!("{msg}");
    }
}

/// Scoped breadcrumb: logs every `stage()` call of the load path with a millisecond
/// stamp, which is how a browser stall gets localised.
pub fn stage(name: &str) -> Stage {
    Stage {
        name: name.to_string(),
        at: crate::clock::now(),
    }
}

/// See [`stage`].
pub struct Stage {
    name: String,
    at: crate::clock::Instant,
}

impl Drop for Stage {
    fn drop(&mut self) {
        log(&format!(
            "[stage] {} took {:.1}s",
            self.name,
            self.at.elapsed().as_secs_f64()
        ));
    }
}
