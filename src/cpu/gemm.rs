//! Blocked f32 GEMM for the CPU encoder: `C[m,n] = A[m,k] @ W[n,k]^T`.
//!
//! Both operands are row-major and `k` is the contiguous axis, so a plain
//! "dot product down W's row" kernel would need 16 strided loads per k step.
//! Instead `W` is repacked once into 16-column panels (`PackW`): for a fixed
//! k the 16 output columns are 64 contiguous bytes, which is exactly what the
//! AVX2 micro-kernel (6 rows x 16 columns) wants. Repacking is cost-free at
//! run time because the encoder weights are constant for the whole decode.
//!
//! Parallelism is over row blocks; every task owns a disjoint output
//! rectangle, so the only unchecked pointer work is inside one task.

// The pointer kernels below are `#[target_feature]` fns whose whole body is a
// single documented unsafe contract; wrapping every intrinsic call in its own
// `unsafe` block would only add noise.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]

use crate::parallel::{SPLIT, num_threads};
use rayon::prelude::*;

/// Output columns per packed panel = one AVX2 register of f32.
const PANEL: usize = 16;
/// Rows of `A` per micro-kernel tile.
const MR: usize = 6;

/// Encoder weights repacked into `[panel][k][PANEL]` order.
///
/// Exactly one of `data` / `data16` is filled. The f16 form halves the resident
/// weight set (2.4 GB -> 1.2 GB) and is numerically identical: the f32 form
/// already holds f16-rounded values (see [`round_to_f16`]), so widening the f16
/// bits back to f32 reproduces the same numbers bit for bit.
pub struct PackW {
    n: usize,
    k: usize,
    data: Vec<f32>,
    data16: Vec<u16>,
    half: bool,
}

/// Whether static weights are stored as f16.
///
/// Default: yes, when the CPU has the F16C kernel; machines without it keep the
/// f32 form and the plain AVX2 / scalar paths, which is what they ran before.
/// `BARK3_CPU_W32=1` forces f32, `BARK3_CPU_W16=1` forces f16 (the two exist so
/// the A/B runs in one binary, where both variants see the same machine).
///
/// Read once: the representation is a process-wide choice.
fn half_weights() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var("BARK3_CPU_W32").is_ok() {
            return false;
        }
        std::env::var("BARK3_CPU_W16").is_ok() || simd16_available()
    })
}

/// Packs a `(n, k)` weight matrix in whichever form the process selected.
///
/// Use this for model weights; activations that are repacked per call (the
/// attention keys/values, the position table) stay on [`PackW::from_rows`].
pub fn pack_rows(w: &[f32], n: usize, k: usize) -> PackW {
    if half_weights() {
        PackW::from_rows_half(w, n, k)
    } else {
        PackW::from_rows(w, n, k)
    }
}

impl PackW {
    pub fn empty() -> Self {
        Self {
            n: 0,
            k: 0,
            data: Vec::new(),
            data16: Vec::new(),
            half: false,
        }
    }

    /// Pack a `(n, k)` row-major weight matrix.
    pub fn from_rows(w: &[f32], n: usize, k: usize) -> Self {
        let mut out = Self::empty();
        out.refill_rows(w, n, k, false);
        out
    }

    /// Pack a `(n, k)` row-major weight matrix as f16.
    pub fn from_rows_half(w: &[f32], n: usize, k: usize) -> Self {
        let mut out = Self::empty();
        out.refill_rows_half(w, n, k);
        out
    }

    /// Pack from a `(k, n)` matrix with row stride `ld`, taking rows
    /// `k_off..k_off + k`. Used for per-layer tensors that arrive transposed
    /// (attention keys/values, relative-position keys).
    pub fn from_cols(src: &[f32], ld: usize, k_off: usize, n: usize, k: usize) -> Self {
        let mut out = Self::empty();
        out.refill_cols(src, ld, k_off, n, k, false);
        out
    }

    #[allow(clippy::needless_range_loop)]
    pub fn refill_rows(&mut self, w: &[f32], n: usize, k: usize, round_f16: bool) {
        self.n = n;
        self.k = k;
        self.half = false;
        self.data16.clear();
        let panels = n.div_ceil(PANEL);
        self.data.clear();
        self.data.resize(panels * k * PANEL, 0.0);
        for p in 0..panels {
            let base = p * k * PANEL;
            for kb in (0..k).step_by(PANEL) {
                let cols = (k - kb).min(PANEL);
                // `tile[jj][j]` is k index `jj`, output column `j`.
                let mut tile = [[0f32; PANEL]; PANEL];
                for j in 0..PANEL {
                    let row = p * PANEL + j;
                    if row >= n {
                        break;
                    }
                    let src = &w[row * k + kb..row * k + kb + cols];
                    for (jj, v) in src.iter().enumerate() {
                        tile[jj][j] = if round_f16 { round_to_f16(*v) } else { *v };
                    }
                }
                for (jj, lane) in tile.iter().enumerate().take(cols) {
                    self.data[base + (kb + jj) * PANEL..][..PANEL].copy_from_slice(lane);
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn refill_rows_half(&mut self, w: &[f32], n: usize, k: usize) {
        self.n = n;
        self.k = k;
        self.half = true;
        self.data.clear();
        let panels = n.div_ceil(PANEL);
        self.data16.clear();
        self.data16.resize(panels * k * PANEL, 0);
        for p in 0..panels {
            let base = p * k * PANEL;
            for kb in (0..k).step_by(PANEL) {
                let cols = (k - kb).min(PANEL);
                let mut tile = [[0u16; PANEL]; PANEL];
                for j in 0..PANEL {
                    let row = p * PANEL + j;
                    if row >= n {
                        break;
                    }
                    let src = &w[row * k + kb..row * k + kb + cols];
                    for (jj, v) in src.iter().enumerate() {
                        tile[jj][j] = half::f16::from_f32(*v).to_bits();
                    }
                }
                for (jj, lane) in tile.iter().enumerate().take(cols) {
                    self.data16[base + (kb + jj) * PANEL..][..PANEL].copy_from_slice(lane);
                }
            }
        }
    }

    pub fn refill_cols(
        &mut self,
        src: &[f32],
        ld: usize,
        k_off: usize,
        n: usize,
        k: usize,
        round_f16: bool,
    ) {
        self.n = n;
        self.k = k;
        self.half = false;
        self.data16.clear();
        let panels = n.div_ceil(PANEL);
        self.data.clear();
        self.data.resize(panels * k * PANEL, 0.0);
        for p in 0..panels {
            let base = p * k * PANEL;
            let cols = (n - p * PANEL).min(PANEL);
            for kb in (0..k).step_by(PANEL) {
                let rows = (k - kb).min(PANEL);
                let mut tile = [[0f32; PANEL]; PANEL];
                for kk in 0..rows {
                    let src_row = &src[(k_off + kb + kk) * ld + p * PANEL..][..cols];
                    for (j, v) in src_row.iter().enumerate() {
                        tile[kk][j] = if round_f16 { round_to_f16(*v) } else { *v };
                    }
                }
                for (kk, lane) in tile.iter().enumerate().take(rows) {
                    self.data[base + (kb + kk) * PANEL..][..PANEL].copy_from_slice(lane);
                }
            }
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }
}

pub fn round_to_f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// `C[0..m, n0..n0+ncols] = A[m,k] @ W[n0..n0+ncols, k]^T`.
///
/// `a` is `(m, k)` row-major, `c` is `(m, ncols)` row-major. `n0` must be a
/// multiple of [`PANEL`]; `ncols` may be anything.
pub fn gemm(a: &[f32], w: &PackW, c: &mut [f32], m: usize, n0: usize, ncols: usize) {
    let k = w.k;
    assert_eq!(n0 % PANEL, 0, "gemm column offset must be panel aligned");
    assert!(n0 + ncols <= w.n, "gemm column range outside the weight");
    assert!(a.len() >= m * k, "gemm A is too short");
    assert!(c.len() >= m * ncols, "gemm C is too short");
    if m == 0 || ncols == 0 || k == 0 {
        return;
    }

    // 2-D tiling of C. A task owns a TM x TN rectangle and walks all of k, so
    // its A slice and W slice stay in L2 while the 6-row micro tiles stream
    // through them. Splitting rows only (the obvious choice) re-reads the whole
    // weight matrix once per row tile and drops small-m shapes to memory speed.
    //
    // Narrow shapes (the per-head attention GEMMs) would only produce a handful
    // of tiles, so the tile shrinks until every core has work.
    // `tn` must stay a multiple of `PANEL`: the packed weights are addressed as
    // `panel = n0 / PANEL`, so a tile that starts mid-panel would read its
    // neighbours' columns. The attention score GEMMs have `ncols = frames`,
    // which is rarely a multiple of 128, and halving 125 gives 62 -> 31 -> a
    // silently wrong result for those frame counts.
    let threads = num_threads();
    // `tm` is kept a multiple of `MR` (the micro-kernel's row tile) so a task
    // never ends on a partial micro tile; the remainder path is vectorised, but
    // a full tile is still the cheaper unit.
    let (mut tm, mut tn) = (
        mr_floor(TM.min(m.max(MR))),
        panel_floor(TN.min(ncols).max(PANEL)),
    );
    loop {
        if m.div_ceil(tm) * ncols.div_ceil(tn) >= threads * 2 {
            break;
        }
        let can_m = tm > MR;
        let can_n = tn > PANEL;
        if !can_m && !can_n {
            break;
        }
        if (can_m && tm >= tn) || !can_n {
            tm = mr_floor((tm / 2).max(MR));
        } else {
            tn = panel_floor(tn / 2);
        }
    }
    debug_assert_eq!(tn % PANEL, 0, "column tile must be panel aligned");
    debug_assert!(
        n0 % PANEL == 0 && (n0 + (ncols.div_ceil(tn) - 1) * tn) % PANEL == 0,
        "every column tile must start on a panel boundary"
    );
    let tasks: Vec<Task> = (0..m.div_ceil(tm))
        .flat_map(|mi| {
            let m0 = mi * tm;
            let rows = (m - m0).min(tm);
            (0..ncols.div_ceil(tn)).map(move |ni| Task {
                m0,
                rows,
                n0: n0 + ni * tn,
                cols: (ncols - ni * tn).min(tn),
            })
        })
        .collect();

    let dst = SendPtr(c.as_mut_ptr());
    let src = SendPtr(a.as_ptr() as *mut f32);
    // Borrow the wrappers so the closure captures `&SendPtr` (which is `Sync`)
    // instead of the bare pointers.
    let (src, dst) = (&src, &dst);
    let run = |t: &Task| {
        // SAFETY: tasks are disjoint `(row, column)` rectangles of `c`, and the
        // assertions above guarantee both buffers cover them.
        unsafe {
            tile(
                src.0.add(t.m0 * k),
                k,
                w,
                t.n0,
                t.cols,
                dst.0.add(t.m0 * ncols + (t.n0 - n0)),
                ncols,
                t.rows,
                0,
                k,
            );
        }
    };
    if SPLIT {
        tasks.par_iter().for_each(run);
    } else {
        tasks.iter().for_each(run);
    }
}

/// Rows / columns of one C tile.
const TM: usize = 192;
const TN: usize = 128;

/// Largest multiple of [`MR`] that does not exceed `v`, floored at one tile.
fn mr_floor(v: usize) -> usize {
    (v / MR * MR).max(MR)
}

/// Largest multiple of [`PANEL`] that does not exceed `v`, floored at one panel.
#[inline]
fn panel_floor(v: usize) -> usize {
    (v / PANEL * PANEL).max(PANEL)
}

/// Raw output pointer shared across the rayon workers; every task writes a
/// disjoint rectangle.
struct SendPtr(*mut f32);
// SAFETY: the rectangles handed to the workers never overlap.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

struct Task {
    m0: usize,
    rows: usize,
    n0: usize,
    cols: usize,
}

/// Split-k variant for short-and-wide products such as the attention AV GEMM
/// (`m` and `n` are both 128 while `k` is the frame count).
///
/// `m x ncols` alone cannot fill the cores, so k is cut into one chunk per
/// worker, each worker keeps the full output tile in registers, and the partial
/// sums are reduced. `parts` must hold `workers * m * ncols` floats.
pub fn gemm_split_k(
    a: &[f32],
    w: &PackW,
    c: &mut [f32],
    parts: &mut [f32],
    m: usize,
    n0: usize,
    ncols: usize,
) {
    let k = w.k;
    let threads = num_threads();
    // Keep at least 64 k values per worker so the tile loop stays meaningful.
    let workers = threads.min(k.div_ceil(64)).max(1);
    if workers <= 1 || parts.len() < workers * m * ncols {
        return gemm(a, w, c, m, n0, ncols);
    }
    let chunk = k.div_ceil(workers);
    let src = SendPtr(a.as_ptr() as *mut f32);
    let dst = SendPtr(parts.as_mut_ptr());
    let (src, dst) = (&src, &dst);
    let run = |wi: usize| {
        let k0 = wi * chunk;
        let k1 = (k0 + chunk).min(k);
        if k0 >= k1 {
            return;
        }
        // SAFETY: worker `wi` writes only `parts[wi*m*ncols ..][..m*ncols]`.
        unsafe {
            tile(
                src.0,
                k,
                w,
                n0,
                ncols,
                dst.0.add(wi * m * ncols),
                ncols,
                m,
                k0,
                k1,
            );
        }
    };
    if SPLIT {
        (0..workers).into_par_iter().for_each(run);
    } else {
        (0..workers).for_each(run);
    }
    let out = &mut c[..m * ncols];
    let reduce = |bi: usize, chunk: &mut [f32]| {
        for (j, v) in chunk.iter_mut().enumerate() {
            let i = bi * 1024 + j;
            let mut acc = 0f32;
            for wi in 0..workers {
                acc += parts[wi * m * ncols + i];
            }
            *v = acc;
        }
    };
    if SPLIT {
        out.par_chunks_mut(1024)
            .enumerate()
            .for_each(|(bi, chunk)| reduce(bi, chunk));
    } else {
        out.chunks_mut(1024)
            .enumerate()
            .for_each(|(bi, chunk)| reduce(bi, chunk));
    }
}

#[inline]
fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The f16 kernels additionally need F16C (Haswell 2013 and later, and every
/// AMD core since Piledriver) to widen eight halves per instruction. Machines
/// without it keep the f32 path.
#[inline]
fn simd16_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| simd_available() && std::arch::is_x86_feature_detected!("f16c"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// One row block against the panel range `n0 .. n0 + ncols`.
///
/// # Safety
/// `a` must point at `rows * lda` readable floats, `c` at `rows * ldc`
/// writable floats, and the column range must lie inside `w`.
unsafe fn tile(
    a: *const f32,
    lda: usize,
    w: &PackW,
    n0: usize,
    ncols: usize,
    c: *mut f32,
    ldc: usize,
    rows: usize,
    k0: usize,
    k1: usize,
) {
    if w.half {
        #[cfg(target_arch = "x86_64")]
        if simd16_available() {
            // SAFETY: the feature test above guarantees avx2/fma/f16c.
            unsafe { tile_avx2_half(a, lda, w, n0, ncols, c, ldc, rows, k0, k1) };
            return;
        }
        unsafe { tile_scalar_half(a, lda, w, n0, ncols, c, ldc, rows, k0, k1) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if simd_available() {
        // SAFETY: the feature test above guarantees both target features.
        unsafe { tile_avx2(a, lda, w, n0, ncols, c, ldc, rows, k0, k1) };
        return;
    }
    unsafe { tile_scalar(a, lda, w, n0, ncols, c, ldc, rows, k0, k1) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2(
    a: *const f32,
    lda: usize,
    w: &PackW,
    n0: usize,
    ncols: usize,
    c: *mut f32,
    ldc: usize,
    rows: usize,
    k0: usize,
    k1: usize,
) {
    use std::arch::x86_64::*;

    let k = w.k;
    let first = n0 / PANEL;
    let panels = ncols.div_ceil(PANEL);

    // One micro tile of `N` rows x 16 columns. `N` is a literal, so the row
    // loop unrolls and every accumulator stays in a register. Rows shorter than
    // MR use the same kernel with their own N instead of falling back to a
    // scalar loop: a scalar remainder used to cost more than the vector body
    // when the tile was small (short utterances, the per-head attention GEMMs).
    macro_rules! micro {
        ($n:literal, $wp:ident, $row:ident, $buf:ident, $($r:literal),+) => {{
            let mut acc = [_mm256_setzero_ps(); $n * 2];
            for kk in k0..k1 {
                let wpk = $wp.add(kk * PANEL);
                let w0 = _mm256_loadu_ps(wpk);
                let w1 = _mm256_loadu_ps(wpk.add(8));
                $(
                    let av = _mm256_broadcast_ss(&*a.add(($row + $r) * lda + kk));
                    acc[$r * 2] = _mm256_fmadd_ps(av, w0, acc[$r * 2]);
                    acc[$r * 2 + 1] = _mm256_fmadd_ps(av, w1, acc[$r * 2 + 1]);
                )+
            }
            $(
                _mm256_storeu_ps($buf.as_mut_ptr().add($r * PANEL), acc[$r * 2]);
                _mm256_storeu_ps($buf.as_mut_ptr().add($r * PANEL + 8), acc[$r * 2 + 1]);
            )+
        }};
    }

    let mut row = 0usize;
    while row < rows {
        let tile_rows = (rows - row).min(MR);
        for p in 0..panels {
            let cols = (ncols - p * PANEL).min(PANEL);
            let wp = w.data.as_ptr().add((first + p) * k * PANEL);
            let mut buf = [0f32; MR * PANEL];
            match tile_rows {
                6 => micro!(6, wp, row, buf, 0, 1, 2, 3, 4, 5),
                5 => micro!(5, wp, row, buf, 0, 1, 2, 3, 4),
                4 => micro!(4, wp, row, buf, 0, 1, 2, 3),
                3 => micro!(3, wp, row, buf, 0, 1, 2),
                2 => micro!(2, wp, row, buf, 0, 1),
                _ => micro!(1, wp, row, buf, 0),
            }
            for r in 0..tile_rows {
                let dst = c.add((row + r) * ldc + p * PANEL);
                for j in 0..cols {
                    *dst.add(j) = buf[r * PANEL + j];
                }
            }
        }
        row += tile_rows;
    }
}

/// f16 twin of [`tile_avx2`]: same tile shape and the same k order, with one
/// `vcvtph2ps` pair replacing the two f32 loads. It is written out rather than
/// templated because the intrinsics only inline inside a `#[target_feature]`
/// function, and this loop is the whole backend's hot path.
///
/// # Safety
/// Same contract as [`tile`], and `w.data16` must be the populated form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_avx2_half(
    a: *const f32,
    lda: usize,
    w: &PackW,
    n0: usize,
    ncols: usize,
    c: *mut f32,
    ldc: usize,
    rows: usize,
    k0: usize,
    k1: usize,
) {
    use std::arch::x86_64::*;

    let k = w.k;
    let first = n0 / PANEL;
    let panels = ncols.div_ceil(PANEL);
    let wbase = w.data16.as_ptr();

    // f16 twin of the f32 `micro!` above: same N-row x 16-column tile, same k
    // order, with one `vcvtph2ps` pair replacing the two f32 loads.
    macro_rules! micro {
        ($n:literal, $wp:ident, $row:ident, $buf:ident, $($r:literal),+) => {{
            let mut acc = [_mm256_setzero_ps(); $n * 2];
            for kk in k0..k1 {
                let wpk = $wp.add(kk * PANEL);
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(wpk as *const __m128i));
                let w1 = _mm256_cvtph_ps(_mm_loadu_si128(wpk.add(8) as *const __m128i));
                $(
                    let av = _mm256_broadcast_ss(&*a.add(($row + $r) * lda + kk));
                    acc[$r * 2] = _mm256_fmadd_ps(av, w0, acc[$r * 2]);
                    acc[$r * 2 + 1] = _mm256_fmadd_ps(av, w1, acc[$r * 2 + 1]);
                )+
            }
            $(
                _mm256_storeu_ps($buf.as_mut_ptr().add($r * PANEL), acc[$r * 2]);
                _mm256_storeu_ps($buf.as_mut_ptr().add($r * PANEL + 8), acc[$r * 2 + 1]);
            )+
        }};
    }

    let mut row = 0usize;
    while row < rows {
        let tile_rows = (rows - row).min(MR);
        for p in 0..panels {
            let cols = (ncols - p * PANEL).min(PANEL);
            let wp = wbase.add((first + p) * k * PANEL);
            let mut buf = [0f32; MR * PANEL];
            match tile_rows {
                6 => micro!(6, wp, row, buf, 0, 1, 2, 3, 4, 5),
                5 => micro!(5, wp, row, buf, 0, 1, 2, 3, 4),
                4 => micro!(4, wp, row, buf, 0, 1, 2, 3),
                3 => micro!(3, wp, row, buf, 0, 1, 2),
                2 => micro!(2, wp, row, buf, 0, 1),
                _ => micro!(1, wp, row, buf, 0),
            }
            for r in 0..tile_rows {
                let dst = c.add((row + r) * ldc + p * PANEL);
                for j in 0..cols {
                    *dst.add(j) = buf[r * PANEL + j];
                }
            }
        }
        row += tile_rows;
    }
}

/// Portable f16 path with the same accumulation order as the vector kernel.
///
/// # Safety
/// Same contract as [`tile`].
unsafe fn tile_scalar_half(
    a: *const f32,
    lda: usize,
    w: &PackW,
    n0: usize,
    ncols: usize,
    c: *mut f32,
    ldc: usize,
    rows: usize,
    k0: usize,
    k1: usize,
) {
    let k = w.k;
    let first = n0 / PANEL;
    let panels = ncols.div_ceil(PANEL);
    for row in 0..rows {
        for p in 0..panels {
            let cols = (ncols - p * PANEL).min(PANEL);
            let wp = w.data16.as_ptr().add((first + p) * k * PANEL);
            for j in 0..cols {
                let mut acc = 0f32;
                for kk in k0..k1 {
                    acc += *a.add(row * lda + kk)
                        * half::f16::from_bits(*wp.add(kk * PANEL + j)).to_f32();
                }
                *c.add(row * ldc + p * PANEL + j) = acc;
            }
        }
    }
}

/// Portable path with the same accumulation order as the vector kernel.
///
/// # Safety
/// Same contract as [`tile`].
unsafe fn tile_scalar(
    a: *const f32,
    lda: usize,
    w: &PackW,
    n0: usize,
    ncols: usize,
    c: *mut f32,
    ldc: usize,
    rows: usize,
    k0: usize,
    k1: usize,
) {
    let k = w.k;
    let first = n0 / PANEL;
    let panels = ncols.div_ceil(PANEL);
    for row in 0..rows {
        for p in 0..panels {
            let cols = (ncols - p * PANEL).min(PANEL);
            let wp = w.data.as_ptr().add((first + p) * k * PANEL);
            for j in 0..cols {
                let mut acc = 0f32;
                for kk in k0..k1 {
                    acc += *a.add(row * lda + kk) * *wp.add(kk * PANEL + j);
                }
                *c.add(row * ldc + p * PANEL + j) = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], w: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for r in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[r * k + kk] * w[j * k + kk];
                }
                c[r * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn matches_naive_for_odd_shapes() {
        for (m, n, k) in [(1, 16, 5), (7, 40, 33), (13, 128, 64), (191, 64, 17)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01 - 0.05).collect();
            let w: Vec<f32> = (0..n * k).map(|i| (i % 13) as f32 * 0.02 - 0.1).collect();
            let pw = PackW::from_rows(&w, n, k);
            let mut c = vec![0f32; m * n];
            gemm(std::hint::black_box(&a), &pw, &mut c, m, 0, n);
            let want = naive(&a, &w, n, k, m);
            let mut max = 0f32;
            for i in 0..m * n {
                max = max.max((c[i] - want[i]).abs());
            }
            assert!(max < 1e-4, "shape {m}x{n}x{k}: max diff {max}");
        }
    }

    /// f16 storage must reproduce the f32 form exactly: the weights are already
    /// f16-rounded before packing, so widening them back is lossless.
    #[test]
    fn half_pack_matches_f32_pack() {
        for (m, n, k) in [(6, 16, 16), (13, 40, 33), (24, 128, 64), (7, 96, 40)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 19) as f32 * 0.013 - 0.11).collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| round_to_f16((i % 23) as f32 * 0.017 - 0.2))
                .collect();
            let pw32 = PackW::from_rows(&w, n, k);
            let pw16 = PackW::from_rows_half(&w, n, k);
            let mut c32 = vec![0f32; m * n];
            let mut c16 = vec![0f32; m * n];
            gemm(&a, &pw32, &mut c32, m, 0, n);
            gemm(&a, &pw16, &mut c16, m, 0, n);
            for i in 0..m * n {
                assert_eq!(
                    c32[i].to_bits(),
                    c16[i].to_bits(),
                    "shape {m}x{n}x{k}: element {i} differs"
                );
            }
        }
    }

    /// Every column tile has to start on a PANEL boundary: the packed weights
    /// are addressed by `n0 / PANEL`, so a tile starting mid-panel silently
    /// reads its neighbour's columns. Attention's score GEMM has
    /// `ncols = frames`, and 125 frames used to pick a 31-column tile.
    #[test]
    fn matches_naive_for_unaligned_column_counts() {
        for ncols in 60..200usize {
            let (m, k) = (ncols, 9usize);
            let a: Vec<f32> = (0..m * k).map(|i| (i % 19) as f32 * 0.03 - 0.2).collect();
            let w: Vec<f32> = (0..ncols * k)
                .map(|i| (i % 23) as f32 * 0.02 - 0.15)
                .collect();
            let pw = PackW::from_rows(&w, ncols, k);
            let mut c = vec![0f32; m * ncols];
            gemm(std::hint::black_box(&a), &pw, &mut c, m, 0, ncols);
            let want = naive(&a, &w, ncols, k, m);
            let mut max = 0f32;
            for i in 0..m * ncols {
                max = max.max((c[i] - want[i]).abs());
            }
            assert!(max < 1e-4, "ncols {ncols}: max diff {max}");
        }
    }

    #[test]
    fn throughput() {
        for (m, n, k) in [
            (2208usize, 4096usize, 1024usize),
            (2240, 4096, 1024),
            (448, 4096, 1024),
            (191, 4096, 1024),
            (191, 1024, 4096),
            (191, 3072, 1024),
        ] {
            let a = vec![0.01f32; m * k];
            let w = vec![0.02f32; n * k];
            let pw = PackW::from_rows(&w, n, k);
            let mut c = vec![0f32; m * n];
            let t = crate::clock::now();
            gemm(std::hint::black_box(&a), &pw, &mut c, m, 0, n);
            let dt = t.elapsed().as_secs_f64();
            println!(
                "gemm {m}x{n}x{k}: {dt:.3}s  {:.2} GFLOPS  threads {}  sum {:.3}",
                2.0 * (m * n * k) as f64 / dt / 1e9,
                rayon::current_num_threads(),
                std::hint::black_box(&c)[..1].iter().sum::<f32>()
            );
        }
    }
}
