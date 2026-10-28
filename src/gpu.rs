use anyhow::{Context, Result, bail};
use std::sync::atomic::{AtomicU64, Ordering};

/// Bytes of GPU buffers this process has allocated, for `BARK3_MEMINFO`.
///
/// wgpu exposes no allocator totals, so every `storage` / `uniform` call adds
/// its size here.  The counter never decreases: it reports the *peak* footprint
/// of our own buffers, which is what a memory budget cares about.
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// A wgpu device plus the queue, adapter info and negotiated limits.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    /// Scratch buffers reused across calls — see Gpu::scratch.
    scratch: std::sync::Mutex<std::collections::HashMap<(&'static str, u64), wgpu::Buffer>>,
}

/// Which device to run on.
///
/// One binary, several devices is the whole point of the port, so the choice is
/// explicit and enumerable rather than "whatever wgpu hands back": a caller can
/// run on the integrated GPU while the discrete one is busy, pin a backend for
/// an A/B, or address the same machine's adapters by index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Runtime {
        api: wgpu::Backend,
        index: usize,
    },
    Cpu,
    Index(usize),
    Name(String),
}

/// The compute runtimes this engine can run on, in the order it tries them.
///
/// `cpu` is our own implementation (see [`DeviceSelector::Cpu`]).  There is
/// deliberately no vendor-specific runtime beyond wgpu's own four: wgpu drives
/// Windows through D3D12 **compute**, not DirectML.
pub const RUNTIMES: &[(&str, wgpu::Backend)] = &[
    ("vulkan", wgpu::Backend::Vulkan),
    ("metal", wgpu::Backend::Metal),
    ("dx12", wgpu::Backend::Dx12),
    ("d3d12", wgpu::Backend::Dx12),
    ("gl", wgpu::Backend::Gl),
    ("webgpu", wgpu::Backend::BrowserWebGpu),
];

impl DeviceSelector {
    /// Parse a CLI-style spec.
    ///
    /// * `auto` — the default policy
    /// * `cpu` — the CPU implementation (currently refuses, see [`Self::Cpu`])
    /// * `<runtime>[:<index>]` — `vulkan`, `vulkan:1`, `dx12`, `metal:0`, `gl`
    /// * `#<n>` / `<n>` — raw index into [`list_devices`]
    /// * anything else — substring of the adapter name
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if s.eq_ignore_ascii_case("gpu") {
            bail!("`gpu` is a backend, not an adapter name — use Backend::Gpu or `--device gpu`");
        }
        if let Some(rest) = s.strip_prefix('#') {
            return Ok(Self::Index(rest.trim().parse().context("device index")?));
        }
        if let Ok(i) = s.parse::<usize>() {
            return Ok(Self::Index(i));
        }
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (n, i.trim().parse().context("runtime device index")?),
            None => (s, 0usize),
        };
        if let Some((_, api)) = RUNTIMES.iter().find(|(n, _)| name.eq_ignore_ascii_case(n)) {
            return Ok(Self::Runtime { api: *api, index });
        }
        Ok(Self::Name(s.to_lowercase()))
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        match self {
            Self::Auto => true,
            Self::Name(n) => info.name.to_lowercase().contains(n),
            Self::Index(_) | Self::Cpu => true,
            Self::Runtime { api, .. } => info.backend == *api,
        }
    }
}

/// Discrete, then integrated, then virtual. A CPU adapter is not a GPU.
fn device_class(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    }
}

fn is_real_gpu(info: &wgpu::AdapterInfo) -> bool {
    device_class(info.device_type) <= 2
}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let class = device_class(info.device_type);
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

/// The runtimes in the order [`rank`] prefers them.
const API_RANK_ORDER: [wgpu::Backend; 5] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
    wgpu::Backend::BrowserWebGpu,
];

/// An instance that only knows about `backends`.
///
/// `Instance::default()` enables every runtime this build has, and creating it
/// is where the loaders of those runtimes get loaded: **127 ms** on this machine
/// against **20 ms** for a Vulkan-only instance (Vulkan's own enumeration is
/// 2.6 ms, D3D12's is 924 ms — both after the instance exists).  Flags and
/// backend options stay the defaults `Instance::default()` uses.
fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    if backends == wgpu::Backends::all() {
        return wgpu::Instance::default();
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

/// Adapters `selector` needs to see, gathered no more widely than that.
///
/// This is startup cost, and on Windows it is not small: a default instance plus
/// a D3D12 enumeration is **~1.05 s**, against 20 ms + 2.6 ms for a Vulkan-only
/// instance — all of it before a single weight is read.  So:
///
/// * a named runtime gets an instance of itself and enumerates only itself;
/// * a raw index or a name substring is defined over the whole list, so those
///   still take the full instance and the full enumeration;
/// * `Auto` walks the runtimes in [`rank`] order — instance and all — and stops
///   as soon as a discrete GPU turns up: a later runtime could then only tie on
///   device class, and ties go to the earlier runtime, so the pick is the one
///   the full enumeration would have made.  A machine whose Vulkan has only an
///   iGPU (or nothing) still walks on, because there D3D12 *can* change the
///   answer.
/// The browser backend cannot enumerate adapters, so ask for one: the page gets
/// the high-performance adapter, then the browser's own fallback (integrated, or
/// software when nothing else exists).
#[cfg(target_arch = "wasm32")]
async fn adapters_for(selector: &DeviceSelector) -> Vec<wgpu::Adapter> {
    let _ = selector;
    let instance = instance_for(wgpu::Backends::BROWSER_WEBGPU);
    let ask = |fallback: bool| wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: fallback,
        compatible_surface: None,
        ..Default::default()
    };
    match instance.request_adapter(&ask(false)).await {
        Ok(adapter) => vec![adapter],
        Err(_) => instance
            .request_adapter(&ask(true))
            .await
            .ok()
            .into_iter()
            .collect(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn adapters_for(selector: &DeviceSelector) -> Vec<wgpu::Adapter> {
    match selector {
        DeviceSelector::Runtime { api, .. } => {
            let b = wgpu::Backends::from(*api);
            instance_for(b).enumerate_adapters(b).await
        }
        DeviceSelector::Index(_) | DeviceSelector::Name(_) => {
            instance_for(wgpu::Backends::all())
                .enumerate_adapters(wgpu::Backends::all())
                .await
        }
        DeviceSelector::Auto => {
            let mut all = Vec::new();
            for api in API_RANK_ORDER {
                let b = wgpu::Backends::from(api);
                let mut found = instance_for(b).enumerate_adapters(b).await;
                let discrete = found
                    .iter()
                    .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                all.append(&mut found);
                if discrete {
                    break;
                }
            }
            all
        }
        DeviceSelector::Cpu => Vec::new(),
    }
}

fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!(
                "#{i} {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One enumerated adapter — everything you need to *choose* a device, and to
/// know whether it can run this engine at all, without creating one.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    /// `max_storage_buffer_binding_size` — the limit the tiling exists for.
    pub max_binding_bytes: u64,
    pub max_workgroup_storage: u32,
    pub subgroup: bool,
    /// The adapter's promised subgroup width range.  Kernels that fold lane xors
    /// need `32..=32`; anything else must take the shared-memory path.
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub timestamps: bool,
    /// PCI ids — information for the listing, never a selector.
    pub vendor_id: u32,
    pub device_id: u32,
}

impl DeviceInfo {
    /// One line, same shape as [`Gpu::describe`].
    pub fn describe(&self) -> String {
        let sg = if !self.subgroup {
            "none".to_string()
        } else if self.subgroup_min == self.subgroup_max {
            self.subgroup_min.to_string()
        } else {
            format!("{}..{}", self.subgroup_min, self.subgroup_max)
        };
        format!(
            "{} ({:?}, {:?}) | {} {} | wg workgroup storage {} B, binding {} MiB, subgroup {sg}",
            self.name,
            self.backend,
            self.device_type,
            self.driver,
            self.driver_info,
            self.max_workgroup_storage,
            self.max_binding_bytes / (1024 * 1024),
        )
    }

    fn from_adapter(a: &wgpu::Adapter) -> Self {
        let i = a.get_info();
        let l = a.limits();
        let f = a.features();
        Self {
            name: i.name,
            backend: i.backend,
            device_type: i.device_type,
            driver: i.driver,
            driver_info: i.driver_info,
            max_binding_bytes: l.max_storage_buffer_binding_size,
            max_workgroup_storage: l.max_compute_workgroup_storage_size,
            subgroup: f.contains(wgpu::Features::SUBGROUP),
            subgroup_min: i.subgroup_min_size,
            subgroup_max: i.subgroup_max_size,
            timestamps: f.contains(wgpu::Features::TIMESTAMP_QUERY),
            vendor_id: i.vendor,
            device_id: i.device,
        }
    }
}

/// Every adapter this instance can see, in wgpu's enumeration order — the order
/// [`DeviceSelector::Index`] indexes into.  Note this is one entry per *(device,
/// graphics API)* pair: the same physical GPU appears once per API it is
/// reachable through.  Use [`list_device_groups`] for the device view.
pub async fn list_devices() -> Vec<DeviceInfo> {
    let instance = wgpu::Instance::default();
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .await
        .iter()
        .map(DeviceInfo::from_adapter)
        .collect()
}

/// One selectable target, named the way [`DeviceSelector::parse`] wants it:
/// `vulkan:0`, `dx12:1`, …  This is the user-facing list — the runtime is the
/// axis, the vendor and device class are *information*, because a card of one
/// vendor is reachable through several runtimes and those are different code
/// paths.
#[derive(Debug, Clone)]
pub struct DeviceTarget {
    /// `<runtime>:<index>`, e.g. `vulkan:1` — feed it back via `--device`.
    pub spec: String,
    pub info: DeviceInfo,
    /// True when [`DeviceSelector::Auto`] would pick this target.
    pub is_default: bool,
}

impl DeviceTarget {
    /// `vulkan:0  NVIDIA P104-100 (NVIDIA, dGPU, driver 572.75) binding 2047 MiB, subgroup 32`
    pub fn describe(&self) -> String {
        let sg = if !self.info.subgroup {
            "no subgroup".to_string()
        } else if self.info.subgroup_min == self.info.subgroup_max {
            format!("subgroup {}", self.info.subgroup_min)
        } else {
            format!(
                "subgroup {}..{}",
                self.info.subgroup_min, self.info.subgroup_max
            )
        };
        format!(
            "{}{:<10} {} ({}, {}, driver {}) binding {} MiB, {sg}",
            if self.is_default { "* " } else { "  " },
            self.spec,
            self.info.name,
            vendor_label(self.info.vendor_id),
            self.info.device_type_str(),
            self.info.driver,
            self.info.max_binding_bytes / (1024 * 1024),
        )
    }
}

impl DeviceInfo {
    fn device_type_str(&self) -> &'static str {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => "dGPU",
            wgpu::DeviceType::IntegratedGpu => "iGPU",
            wgpu::DeviceType::VirtualGpu => "vGPU",
            wgpu::DeviceType::Cpu => "CPU",
            _ => "other",
        }
    }
}

fn vendor_label(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        0x1414 => "Microsoft",
        _ => "other",
    }
}

/// The user-facing device list: every adapter, named `<runtime>:<index>`, with
/// the runtime's devices ordered discrete-before-integrated (so the index is
/// stable), and the whole list ordered by [`rank`] with the default marked.
pub async fn list_targets() -> Vec<DeviceTarget> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let mut infos: Vec<DeviceInfo> = adapters.iter().map(DeviceInfo::from_adapter).collect();
    infos.sort_by_key(|d| (rank_of(d), d.name.clone()));

    let mut per_runtime: std::collections::HashMap<wgpu::Backend, usize> =
        std::collections::HashMap::new();
    let mut targets: Vec<DeviceTarget> = Vec::new();
    for info in infos {
        let api = info.backend;
        let n = *per_runtime.entry(api).or_insert(0);
        per_runtime.insert(api, n + 1);
        targets.push(DeviceTarget {
            spec: format!("{}:{n}", runtime_name(api)),
            info,
            is_default: false,
        });
    }
    if let Some(first) = targets.first_mut() {
        first.is_default = true;
    }
    targets
}

/// The name users type for a runtime (and what `warn`/`skip` messages say).
pub fn runtime_name(api: wgpu::Backend) -> &'static str {
    RUNTIMES
        .iter()
        .find(|(_, b)| *b == api)
        .map(|(n, _)| *n)
        .unwrap_or("other")
}

fn rank_of(d: &DeviceInfo) -> (u8, u8) {
    let class = device_class(d.device_type);
    let api = match d.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

impl Gpu {
    /// Enumerate adapters and pick one.  `prefer` matches a case-insensitive
    /// substring of the adapter name (e.g. `"nvidia"`, `"intel"`); without it
    /// Auto tries discrete, then integrated, then a virtual GPU, and skips CPU adapters.
    ///
    /// Shorthand for [`Gpu::new_with`] with [`DeviceSelector::Name`] / `Auto`.
    pub async fn new(prefer: Option<&str>) -> Result<Self> {
        let sel = match prefer {
            Some(p) => DeviceSelector::parse(p)?,
            None => DeviceSelector::Auto,
        };
        Self::new_with(sel).await
    }

    /// The explicit form: choose a runtime (`vulkan`, `dx12`, `metal`, `gl`) and
    /// which device of it, or a raw index / name.
    ///
    /// Shorthand: `Gpu::new(Some("vulkan:1"))`.
    pub async fn new_with(selector: DeviceSelector) -> Result<Self> {
        if selector == DeviceSelector::Cpu {
            bail!(
                "DeviceSelector::Cpu is the host backend, not a wgpu device — \
                 use WgpuAsr::load_on(.., DeviceSelector::Cpu) or `transcribe --cpu-dec`"
            );
        }
        let adapters = adapters_for(&selector).await;
        crate::browser::log(&format!(
            "[gpu] selector {selector:?}: {} adapter(s) {}",
            adapters.len(),
            list_names(&adapters)
        ));
        if adapters.is_empty() {
            bail!("no wgpu adapters found (selector {selector:?}); try listing them first");
        }

        // `Auto` walks every real GPU, discrete before integrated. A named
        // adapter or a runtime index stays on that one device.
        let candidates: Vec<&wgpu::Adapter> = match &selector {
            DeviceSelector::Index(i) => vec![adapters.get(*i).ok_or_else(|| {
                anyhow::anyhow!(
                    "device #{i} does not exist ({} adapter(s) visible: {})",
                    adapters.len(),
                    list_names(&adapters)
                )
            })?],
            sel => {
                let mut hits: Vec<&wgpu::Adapter> = adapters
                    .iter()
                    .filter(|a| sel.matches(&a.get_info()))
                    .filter(|a| {
                        !matches!(sel, DeviceSelector::Auto)
                            || cfg!(target_arch = "wasm32")
                            || is_real_gpu(&a.get_info())
                    })
                    .collect();
                if hits.is_empty() {
                    let hint = if matches!(sel, DeviceSelector::Auto) {
                        String::new()
                    } else {
                        format!(" matching {sel:?}")
                    };
                    bail!("no adapter{hint} (visible: {})", list_names(&adapters));
                }
                hits.sort_by_key(|a| rank(&a.get_info()));
                match sel {
                    DeviceSelector::Runtime { index, .. } => {
                        vec![hits.get(*index).copied().ok_or_else(|| {
                            anyhow::anyhow!(
                                "that runtime has {} device(s), index {index} is out of range",
                                hits.len()
                            )
                        })?]
                    }
                    DeviceSelector::Auto => hits,
                    _ => vec![hits[0]],
                }
            }
        };

        let mut opened = None;
        let mut skipped: Option<anyhow::Error> = None;
        for adapter in &candidates {
            let info = adapter.get_info();
            let features = adapter.features();
            let limits = adapter.limits();
            match adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("bark3-wgpu"),
                    required_features: features
                        & (wgpu::Features::TIMESTAMP_QUERY
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                            | wgpu::Features::SUBGROUP),
                    required_limits: limits.clone(),
                    ..Default::default()
                })
                .await
            {
                Ok((device, queue)) => {
                    crate::browser::log(&format!(
                        "[gpu] device opened: {} ({:?}, {:?})",
                        info.name, info.backend, info.device_type
                    ));
                    opened = Some((info, features, limits, device, queue));
                    break;
                }
                Err(e) => {
                    if candidates.len() > 1 {
                        eprintln!(
                            "qwen3-asr: skip {} ({:?}, {:?}): {e:#}",
                            info.name, info.backend, info.device_type
                        );
                    }
                    skipped = Some(e.into());
                }
            }
        }
        let Some((info, features, limits, device, queue)) = opened else {
            return Err(skipped.context("request_device")?);
        };

        // The browser has no stderr: route device errors to the console or the
        // first broken pipeline would fail silently.
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            crate::browser::log(&format!("[wgpu uncaptured error] {e}"));
        }));

        Ok(Self {
            device,
            queue,
            info,
            limits,
            features,
            scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// One-line description for logs and reports.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {} | wg workgroup storage {} B, maxStorageBufferBindingSize {} MiB",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.limits.max_compute_workgroup_storage_size,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024),
        )
    }

    /// Storage buffer sized `bytes` rounded up to a 16-byte multiple.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        ALLOC_BYTES.fetch_add(size.max(16), Ordering::Relaxed);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// One scratch buffer per label, reused across calls.
    ///
    /// A 60 s chunk allocates about twenty activation buffers (~160 MB). Creating
    /// and dropping those every chunk is cheap on a native driver but not in a
    /// browser: the old buffer only goes back to the driver once the GPU work that
    /// used it has finished *and* the JavaScript object has been collected, so a
    /// long recording walks VRAM up instead of staying flat — on an 8 GB card a
    /// 63 min file reached the limit.
    ///
    /// Reusing by label keeps the steady state at one chunk's worth. A different
    /// size for the same label replaces the entry, so a run whose chunks grow does
    /// not accumulate one buffer per size. A reused buffer still holds the previous
    /// call's contents: every caller must write the whole range it reads.
    pub fn scratch(&self, label: &'static str, bytes: u64) -> wgpu::Buffer {
        let size = ((bytes + 15) & !15).max(16);
        let mut pool = self.scratch.lock().expect("scratch pool");
        // High-water reuse: a chunk's span varies by a second or two and the
        // buffer is sized from it, so an exact-size match would reallocate nearly
        // every chunk. Chrome then carves a fresh device allocation per size and
        // keeps the old ones in its free list: the pool total stayed at 0.5 GB
        // while the card climbed past 5 GB. Keeping the largest buffer per label
        // turns that into a one-time allocation.
        if let Some(((_, have), buf)) = pool.iter().find(|((l, _), _)| *l == label) {
            if *have >= size {
                return buf.clone();
            }
        }
        pool.retain(|(l, _), _| *l != label);
        let buf = self.storage(label, size);
        pool.insert((label, size), buf.clone());
        // A chunk's activations are tens of MB; anything near a gigabyte means a
        // span slipped through the VAD limit, which is worth seeing in the console.
        if size >= (64 << 20) {
            let total: u64 = pool.iter().map(|((_, s), _)| *s).sum();
            crate::browser::log(&format!(
                "[pool] {label} {} MB (pool total {} MB)",
                size >> 20,
                total >> 20
            ));
        }
        buf
    }
    /// Total GPU buffer bytes allocated so far — see [`ALLOC_BYTES`].
    pub fn allocated_bytes(&self) -> u64 {
        ALLOC_BYTES.load(Ordering::Relaxed)
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        ALLOC_BYTES.fetch_add(size.max(16), Ordering::Relaxed);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Runtime-sized deferred write (queue ordering makes it visible to work
    /// submitted afterwards).  For multi-megabyte *load-time* transfers use
    /// [`Gpu::uploader`] instead — deferred staging must be bounded.
    pub fn upload(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    /// [`Gpu::upload`] at an explicit offset — for uniform buffers carrying one
    /// cfg per dispatch slot (see `decoder::prefill`'s slab cfgs).
    pub fn write_at(&self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buf, offset, data);
    }

    /// Wait for everything already submitted.
    ///
    /// A browser cannot block on its device, and it does not have to: WebGPU
    /// submits in queue order, so work added later already sees earlier work.
    /// On the web this is a no-op, which is also why `create()` there must not
    /// rely on polling to surface validation errors.
    pub fn wait(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Submit an empty command buffer and wait for the queue to drain.  This
    /// retires every deferred `write_buffer` copy and frees their staging.
    pub fn flush(&self) -> Result<()> {
        self.pump()?;
        self.wait();
        Ok(())
    }

    /// Submit without waiting.
    ///
    /// `queue.write_buffer` only *stages* its copies; they do not start moving
    /// until the next submission.  A load that stages a whole model and submits
    /// once therefore leaves the bus idle through every conversion and every
    /// memcpy and then pays the entire transfer at the end — the pump is what
    /// lets the DMA of one chunk run against the host preparing the next.
    pub fn pump(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Start a bulk upload session (load-time weight transfer).
    pub fn uploader(&self) -> BulkUpload<'_> {
        BulkUpload {
            gpu: self,
            pending: 0,
        }
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);
        self.map_read(&staging, bytes)
    }

    /// Readback the browser can actually perform.
    ///
    /// `map_async` resolves through the JS event loop, so the browser build has
    /// to await it; a synchronous `readback` would deadlock the page. Native
    /// builds have nothing to wait for beyond the map itself.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn readback_async(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        self.readback(buf, bytes)
    }

    /// Readback the browser can actually perform. See the native version.
    #[cfg(target_arch = "wasm32")]
    pub async fn readback_async(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let (resolve, reject) = (std::rc::Rc::new(resolve), std::rc::Rc::new(reject));
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let done = match r {
                    Ok(()) => resolve.call0(&wasm_bindgen::JsValue::NULL),
                    Err(e) => reject.call1(
                        &wasm_bindgen::JsValue::NULL,
                        &wasm_bindgen::JsValue::from_str(&format!("{e}")),
                    ),
                };
                if let Err(e) = done {
                    web_sys::console::log_1(&format!("map_async settle: {e:?}").into());
                }
            });
        });
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| anyhow::anyhow!("awaiting map: {e:?}"))?;
        let mapped = slice.get_mapped_range()?;
        let mut data = mapped.to_vec();
        drop(mapped);
        staging.unmap();
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Maps an already-populated `MAP_READ` staging buffer and copies it out.
    /// The caller owns the copy; this only waits for the queue and maps.
    pub fn map_read(&self, staging: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for readback")?;
        rx.recv()
            .context("map callback dropped")?
            .context("map buffer")?;
        let mut data = slice.get_mapped_range()?.to_vec();
        staging.unmap();
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Compile a WGSL module + compute pipeline, surfacing validation errors
    /// instead of letting them turn into an opaque panic later.  `layout`
    /// attaches an explicit pipeline layout — required whenever one bind group
    /// is shared across other pipelines, because wgpu's implicit layouts are
    /// pipeline-exclusive.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        // Native can block on the validation scope, which names the pipeline.
        // The browser cannot: error scopes are a stack and popping them out of
        // order panics inside wgpu, so there the validation error is left to
        // `on_uncaptured_error`, which logs it to the console.
        #[cfg(not(target_arch = "wasm32"))]
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipe = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(e) = pollster::block_on(guard.pop()) {
            bail!("pipeline {label} failed validation: {e}");
        }
        Ok(pipe)
    }
}

const STAGING_BUDGET: u64 = 256 << 20;

/// Load-time upload session with bounded outstanding staging.
///
/// Route every model-load transfer through one of these (`Gpu::uploader`);
/// runtime-sized writes (rope tables, token slots, KV seeding) stay on the
/// deferred [`Gpu::upload`] path, where queue ordering already guarantees
/// visibility and the volumes are tiny.
pub struct BulkUpload<'a> {
    gpu: &'a Gpu,
    pending: u64,
}

impl<'a> BulkUpload<'a> {
    /// Allocate a 16B-padded storage buffer (as [`Gpu::storage`]).
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.storage(label, bytes)
    }

    /// Allocate a uniform buffer (as [`Gpu::uniform`]).
    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.uniform(label, bytes)
    }

    /// Stage `data` into `buf`.  Pieces larger than the budget are split so a
    /// single staging buffer can never exceed it.
    ///
    /// Every [`STAGING_BUDGET`] bytes the staged copies are *submitted* — not
    /// waited on — so their DMA runs while the caller converts and stages the
    /// next chunk.  Without that the whole transfer lands on the final
    /// [`Self::finish`]: measured, that is 3.9 s against 3.3 s for the same
    /// 0.6 B model, because the bus sits idle through every conversion and
    /// every staging memcpy.  The budget is what bounds how much can be in
    /// flight.
    pub fn upload(&mut self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        self.upload_at(buf, 0, data)
    }

    /// [`Self::upload`] at an explicit offset — so a fused matrix (`q|k|v`,
    /// `gate|up`) can be written part by part instead of being concatenated on
    /// the host first.  The offset must be a multiple of 4 (wgpu's copy
    /// alignment); every f16 row is a multiple of 2, and every call site here
    /// passes whole matrices.
    pub fn upload_at(&mut self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        anyhow::ensure!(
            offset % 4 == 0,
            "upload offset {offset} is not 4-byte aligned"
        );
        let budget = (STAGING_BUDGET as usize).max(1);
        let mut off = offset;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > STAGING_BUDGET {
                self.pump()?;
            }
            self.gpu.queue.write_buffer(buf, off, piece);
            self.pending += piece.len() as u64;
            off += piece.len() as u64;
        }
        Ok(())
    }

    /// Allocate a storage buffer and fill it from `pieces`, each `(offset, bytes)`.
    ///
    /// The pieces come straight out of [`crate::weights`] in the layout the
    /// kernels want, so a fused matrix is never materialised twice: an f16
    /// checkpoint contributes its own mapped bytes, a bf16 one a single
    /// narrowing pass into the destination layout.
    pub fn upload_pieces(
        &mut self,
        label: &str,
        pieces: &[(u64, bytes::Bytes)],
    ) -> Result<wgpu::Buffer> {
        let total: u64 = pieces.iter().map(|(_, b)| b.len() as u64).sum();
        let buf = self.storage(label, total);
        for (off, data) in pieces {
            self.upload_at(&buf, *off, data)?;
        }
        Ok(buf)
    }

    /// Retire whatever is still staged.  Call before the first real dispatch.
    pub fn finish(mut self) -> Result<()> {
        if self.pending > 0 {
            self.flush()?;
        }
        Ok(())
    }

    /// Submit the staged copies without waiting for them.
    fn pump(&mut self) -> Result<()> {
        self.gpu.pump()?;
        self.pending = 0;
        Ok(())
    }

    /// Submit and wait — the last one, where the transfer really is the wait.
    fn flush(&mut self) -> Result<()> {
        self.gpu.flush()?;
        self.pending = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_orders_discrete_before_integrated() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, VirtualGpu};
        assert!(device_class(DiscreteGpu) < device_class(IntegratedGpu));
        assert!(device_class(IntegratedGpu) < device_class(VirtualGpu));
        assert!(device_class(VirtualGpu) < device_class(Cpu));
    }

    #[test]
    fn gpu_is_a_backend_not_an_adapter_name() {
        assert!(DeviceSelector::parse("gpu").is_err());
        assert_eq!(DeviceSelector::parse("auto").unwrap(), DeviceSelector::Auto);
        assert_eq!(DeviceSelector::parse("cpu").unwrap(), DeviceSelector::Cpu);
        assert_eq!(
            DeviceSelector::parse("vulkan:1").unwrap(),
            DeviceSelector::Runtime {
                api: wgpu::Backend::Vulkan,
                index: 1
            }
        );
    }
}
