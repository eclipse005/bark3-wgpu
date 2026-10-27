//! WGSL sources for the small elementwise / reduction kernels that run
//! between GEMMs (LN, silu, GLU, softmax, head packing, dwconv, ...).
//!
//! The raw strings are handed to naga verbatim, so their contents are part of
//! the program's observable behaviour: do not reformat or re-wrap them.

/// rel_shift formula: bd_shifted[q, k] = bd[q, (T-1-q) + k]  (verified vs HF).
pub(super) const P_GLU: &str = r#"
struct Cfg { c2: u32, total2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
const LOG2E: f32 = 1.4426950408889634;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.total2) { return; }
    let row = i / cfg.c2;
    let c = i - row * cfg.c2;
    let a = unpack2x16float(X[row * cfg.c2 * 2u + c]);
    let b = unpack2x16float(X[row * cfg.c2 * 2u + cfg.c2 + c]);
    Out[i] = pack2x16float(vec2<f32>(
        a.x / (1.0 + exp2(-b.x * LOG2E)),
        a.y / (1.0 + exp2(-b.y * LOG2E))));
}
"#;

/// q (T,1024) + bias_u/bias_v (8,128) -> q_u, q_v head-packed (8, T, 128).
/// Input is a column slice (col_off) of a (T, stride) buffer. One thread per word.
pub(super) const P_ADD_HEAD: &str = r#"
struct Cfg { t: u32, words: u32, stride2: u32, coff2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       Q:  array<u32>;
@group(0) @binding(1) var<storage, read>       U:  array<u32>;
@group(0) @binding(2) var<storage, read>       V:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Qu: array<u32>;
@group(0) @binding(4) var<storage, read_write> Qv: array<u32>;
@group(0) @binding(5) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;                       // word index over the (T, 1024) slice
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;                    // j = h*64 + d/2
    let h = j / 64u;
    let qv = unpack2x16float(Q[t * cfg.stride2 + j + cfg.coff2]);
    let uv = unpack2x16float(U[j]);
    let vv = unpack2x16float(V[j]);
    let ow = (h * cfg.t + t) * 64u + (j % 64u);
    Qu[ow] = pack2x16float(vec2<f32>(qv.x + uv.x, qv.y + uv.y));
    Qv[ow] = pack2x16float(vec2<f32>(qv.x + vv.x, qv.y + vv.y));
}
"#;

/// flat (T,1024) -> heads (8, T, 128); input may be a slice of a wider buffer.
pub(super) const P_FLAT_TO_HEADS: &str = r#"
struct Cfg { t: u32, words: u32, stride2: u32, coff2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let ow = (h * cfg.t + t) * 64u + (j % 64u);
    Out[ow] = X[t * cfg.stride2 + j + cfg.coff2];
}
"#;

/// heads (8, T, 128) -> flat (T, 1024).
pub(super) const P_HEADS_TO_FLAT: &str = r#"
struct Cfg { t: u32, words: u32, _s: u32, _o: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;                       // output word index over (T, 1024)
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let iw = (h * cfg.t + t) * 64u + (j % 64u);
    Out[w] = X[iw];
}
"#;

/// scores[h,q,k] = scale * (content[h,q,k] + bd[h,q, (T-1-q)+k]); cols >= n_real of
/// content/bd may be garbage; we clamp k to real T for reads (bd p <= 2T-2 real).
pub(super) const P_REL_ADD: &str = r#"
struct Cfg { t_real: u32, t_pad: u32, p_stride: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       Content: array<u32>;
@group(0) @binding(1) var<storage, read>       Bd:      array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // one thread per word; 2D grid with x fixed at 65535 groups when needed
    let w = gid.x + gid.y * 16776960u;   // word index over (8, t_pad, t_pad) f16
    let tp = cfg.t_pad;
    let words = 8u * tp * tp / 2u;
    if (w >= words) { return; }
    let k = (w % (tp / 2u)) * 2u;
    let q = (w / (tp / 2u)) % tp;
    let h = w / (tp * tp / 2u);
    let cv = unpack2x16float(Content[w]);
    var v0 = 0.0; var v1 = 0.0;
    if (q < cfg.t_real && k + 1u < cfg.t_real) {
        let p0 = (cfg.t_real - 1u - q) + k;
        let bde0 = (h * tp + q) * cfg.p_stride + p0;
        let bv = unpack2x16float(Bd[bde0 / 2u]);
        v0 = cfg.scale * (cv.x + bv.x);
        v1 = cfg.scale * (cv.y + bv.y);
    } else if (q < cfg.t_real && k < cfg.t_real) {
        let p0 = (cfg.t_real - 1u - q) + k;
        let bde0 = (h * tp + q) * cfg.p_stride + p0;
        let bv = unpack2x16float(Bd[bde0 / 2u]);
        v0 = cfg.scale * (cv.x + bv.x);
        v1 = 0.0;
    }
    Out[w] = pack2x16float(vec2<f32>(v0, v1));
}
"#;

/// softmax rows of width up to 8192: one workgroup per row, 256 threads.
pub(super) const P_SOFTMAX: &str = r#"
struct Cfg { rows: u32, n: u32, stride: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
var<workgroup> red_max: array<f32, 256>;
var<workgroup> red_sum: array<f32, 256>;
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    if (row >= cfg.rows) { return; }
    let basew = row * (cfg.stride / 2u);   // WORD base of this row
    // The most negative f32, written as a decimal that Tint accepts: it rejects
    // any literal above f32::MAX instead of rounding it (3.4028235e38 lands on
    // the same f32 but is refused), and it folds bitcast<f32>(0xFF800000u) into
    // an -inf constant, which WGSL cannot spell. Indistinguishable from -inf for
    // a max accumulator over finite input.
    var m = -3.4028234e38;
    for (var j = lid.x; j < cfg.n; j = j + 256u) {
        let v = half_at(unpack2x16float(X[basew + (j >> 1u)]), j);
        if (v > m) { m = v; }
    }
    red_max[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }
        workgroupBarrier();
    }
    let row_max = red_max[0];
    workgroupBarrier();
    var acc = 0.0;
    for (var j = lid.x; j < cfg.n; j = j + 256u) {
        let v = half_at(unpack2x16float(X[basew + (j >> 1u)]), j);
        acc = acc + exp(v - row_max);
    }
    red_sum[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }
        workgroupBarrier();
    }
    let sum = red_sum[0];
    workgroupBarrier();
    for (var j2 = lid.x; j2 < cfg.stride / 2u; j2 = j2 + 256u) {
        let j = j2 * 2u;
        var o0 = 0.0; var o1 = 0.0;
        let v = unpack2x16float(X[basew + j2]);
        if (j < cfg.n) { o0 = exp(v.x - row_max) / sum; }
        if (j + 1u < cfg.n) { o1 = exp(v.y - row_max) / sum; }
        Out[basew + j2] = pack2x16float(vec2<f32>(o0, o1));
    }
}
"#;

/// LayerNorm, f32 in/out, params f32 (w[1024] then b[1024]).
pub(super) const P_LN_F32: &str = r#"
struct LnCfg { last: u32, outer: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       Src: array<f32>;
@group(0) @binding(1) var<storage, read>       Wgt: array<f32>;
@group(0) @binding(2) var<storage, read>       Bia: array<f32>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<f32>;
var<workgroup> red_s: array<f32, 256>;
var<workgroup> red_q: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    if (row >= cfg.outer) { return; }
    let base = row * cfg.last;
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
    var q0 = 0.0; var q1 = 0.0; var q2 = 0.0; var q3 = 0.0;
    let e0 = Src[base + lid.x];
    let e1 = Src[base + lid.x + 256u];
    let e2 = Src[base + lid.x + 512u];
    let e3 = Src[base + lid.x + 768u];
    s0 = e0; s1 = e1; s2 = e2; s3 = e3;
    q0 = e0 * e0; q1 = e1 * e1; q2 = e2 * e2; q3 = e3 * e3;
    let sL = (s0 + s2) + (s1 + s3);
    let qL = (q0 + q2) + (q1 + q3);
    red_s[lid.x] = sL;
    red_q[lid.x] = qL;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            red_s[lid.x] = red_s[lid.x] + red_s[lid.x + s];
            red_q[lid.x] = red_q[lid.x] + red_q[lid.x + s];
        }
        workgroupBarrier();
    }
    let mean = red_s[0] / f32(cfg.last);
    let var_ = red_q[0] / f32(cfg.last) - mean * mean;
    let inv_std = inverseSqrt(var_ + cfg.eps);
    workgroupBarrier();
    for (var w = lid.x; w < cfg.last; w = w + 256u) {
        Dst[base + w] = (Src[base + w] - mean) * inv_std * Wgt[w] + Bia[w];
    }
}
"#;

/// silu in place, f32.
pub(super) const P_SILU_F32: &str = r#"
struct NCfg { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> cfg: NCfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let v = X[i];
    X[i] = v / (1.0 + exp(-v));
}
"#;

/// GLU halves on (T, 2048) f32 -> (T, 1024) f32.
pub(super) const P_GLU_F32: &str = r#"
struct Cfg { c: u32, total: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.total) { return; }
    let row = i / cfg.c;
    let c = i - row * cfg.c;
    let a = X[row * cfg.c * 2u + c];
    let b = X[row * cfg.c * 2u + cfg.c + c];
    Out[i] = a / (1.0 + exp(-b));
}
"#;

/// A[i] += scale * B[i], f32.
pub(super) const P_ADD_SCALED_F32: &str = r#"
struct Cfg { n: u32, scale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    A[i] = A[i] + cfg.scale * B[i];
}
"#;

/// flat f32 slice of (T, stride) -> heads (8, T, 128) f16 (for W-side k/v/relk).
pub(super) const P_F2H_MIX: &str = r#"
struct Cfg { t: u32, words: u32, stride2: u32, coff2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;   // f16-pair index over the (T, 1024) slice
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let base = t * cfg.stride2 + j * 2u + cfg.coff2;   // f32 word base of the pair (stride2 = full stride, coff2 = word offset)
    let a = X[base];
    let b = X[base + 1u];
    let ow = (h * cfg.t + t) * 64u + (j % 64u);
    Out[ow] = pack2x16float(vec2<f32>(a, b));
}
"#;

/// add_head reading f32 qkv, writing qu/qv f32.
pub(super) const P_ADD_HEAD_F32B: &str = r#"
struct Cfg { t: u32, words: u32, stride2: u32, coff2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       Q:  array<f32>;
@group(0) @binding(1) var<storage, read>       U:  array<f32>;
@group(0) @binding(2) var<storage, read>       V:  array<f32>;
@group(0) @binding(3) var<storage, read_write> Qu: array<f32>;
@group(0) @binding(4) var<storage, read_write> Qv: array<f32>;
@group(0) @binding(5) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;   // f32-pair index over the (T, 1024) slice
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let base = t * cfg.stride2 + j * 2u + cfg.coff2;
    let q0 = Q[base];
    let q1 = Q[base + 1u];
    let ow = (h * cfg.t + t) * 128u + (j % 64u) * 2u;
    Qu[ow] = q0 + U[j * 2u];
    Qu[ow + 1u] = q1 + U[j * 2u + 1u];
    Qv[ow] = q0 + V[j * 2u];
    Qv[ow + 1u] = q1 + V[j * 2u + 1u];
}
"#;

/// heads (8, T, 128) f32 -> flat (T, 1024) f32.
pub(super) const P_H2F_F32B: &str = r#"
struct Cfg { t: u32, words: u32, _s: u32, _o: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;   // f32-pair index over output (T, 1024)
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let iw = (h * cfg.t + t) * 128u + (j % 64u) * 2u;
    Out[t * 1024u + j * 2u] = X[iw];
    Out[t * 1024u + j * 2u + 1u] = X[iw + 1u];
}
"#;

/// depthwise conv k=9, X/Out f32, W (1024, 9) f16.
pub(super) const P_DWCONV9_F32: &str = r#"
struct Cfg { t: u32, words: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;   // f32 word = (t, c pair)
    if (w >= cfg.words) { return; }
    let c = (w % 512u) * 2u;
    let t = w / 512u;
    var acc0 = 0.0; var acc1 = 0.0;
    for (var i = 0u; i < 9u; i = i + 1u) {
        var tt = t + i;
        if (tt < 4u) { continue; }
        tt = tt - 4u;
        if (tt >= cfg.t) { continue; }
        let xw = tt * 512u + (w % 512u);
        let xv0 = X[xw * 2u];
        let xv1 = X[xw * 2u + 1u];
        let w0 = c * 9u + i;
        let w1 = (c + 1u) * 9u + i;
        let wv0 = half_at(unpack2x16float(W[w0 / 2u]), w0);
        let wv1 = half_at(unpack2x16float(W[w1 / 2u]), w1);
        acc0 = acc0 + xv0 * wv0;
        acc1 = acc1 + xv1 * wv1;
    }
    Out[w * 2u] = acc0;
    Out[w * 2u + 1u] = acc1;
}
"#;

/// BatchNorm+SiLU, X/Out f32, AB f32.
pub(super) const P_BN_SILU_F32: &str = r#"
struct Cfg { t: u32, words: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       AB: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.words * 2u) { return; }
    let c = i % 1024u;
    let y = X[i] * AB[c] + AB[1024u + c];
    Out[i] = y / (1.0 + exp(-y));
}
"#;

/// X[row, last] += bias[last], f32 broadcast.
pub(super) const P_ADD_BIAS_F32: &str = r#"
struct BiasCfg { outer: u32, last: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: BiasCfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.outer * cfg.last) { return; }
    X[i] = X[i] + B[i % cfg.last];
}
"#;

/// q_u/q_v head-packed as f32.
pub(super) const P_ADD_HEAD_F32: &str = r#"
struct Cfg { t: u32, words: u32, stride2: u32, coff2: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       Q:  array<u32>;
@group(0) @binding(1) var<storage, read>       U:  array<u32>;
@group(0) @binding(2) var<storage, read>       V:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Qu: array<f32>;
@group(0) @binding(4) var<storage, read_write> Qv: array<f32>;
@group(0) @binding(5) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let qv = unpack2x16float(Q[t * cfg.stride2 + j + cfg.coff2]);
    let uv = unpack2x16float(U[j]);
    let vv = unpack2x16float(V[j]);
    let ow = (h * cfg.t + t) * 128u + (j % 64u) * 2u;
    Qu[ow] = qv.x + uv.x;
    Qu[ow + 1u] = qv.y + uv.y;
    Qv[ow] = qv.x + vv.x;
    Qv[ow + 1u] = qv.y + vv.y;
}
"#;

/// rel_add + scale, f32 in/out.
pub(super) const P_REL_ADD_F32: &str = r#"
struct Cfg { t_real: u32, t_pad: u32, p_stride: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       Content: array<f32>;
@group(0) @binding(1) var<storage, read>       Bd:      array<f32>;
@group(0) @binding(2) var<storage, read_write> Out:     array<f32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x + gid.y * 16776960u;
    let tp = cfg.t_pad;
    let words = 8u * tp * tp;
    if (w >= words) { return; }
    let k = w % tp;
    let q = (w / tp) % tp;
    let c = Content[w];
    var v = 0.0;
    if (q < cfg.t_real && k < cfg.t_real) {
        let p = (cfg.t_real - 1u - q) + k;
        v = cfg.scale * (c + Bd[(w / (tp * tp)) * cfg.t_pad * cfg.p_stride + q * cfg.p_stride + p]);
    }
    Out[w] = v;
}
"#;

/// `rel_add` + `softmax` in one pass over the score matrix.
///
/// The two kernels it replaces moved the `(8, T, T)` scores three times
/// (rel_add read+write, softmax max/sum/store reads, softmax write) at DRAM
/// speed. Here one workgroup owns one attention row: it reads the scores and the
/// relative-position window once, keeps the biased row in workgroup memory, and
/// writes the probabilities once. Same values, same reduction order, so the
/// output is bit-identical to the two-kernel sequence.
///
/// `cache` bounds the row length; longer utterances fall back to the pair (see
/// `MAX_ROW` in `encoder.rs`, kept in sync with this array).
pub(super) const P_ADD_SOFTMAX_F32: &str = r#"
struct Cfg { t_real: u32, t_pad: u32, p_stride: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       Content: array<f32>;
@group(0) @binding(1) var<storage, read>       Bd:      array<f32>;
@group(0) @binding(2) var<storage, read_write> Out:     array<f32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
var<workgroup> cache: array<f32, 4096>;
var<workgroup> red_max: array<f32, 256>;
var<workgroup> red_sum: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tp = cfg.t_pad;
    if (row >= 8u * tp) { return; }
    let q = row % tp;
    let base = row * tp;
    if (q >= cfg.t_real) {
        for (var j = lid.x; j < tp; j = j + 256u) { Out[base + j] = 0.0; }
        return;
    }
    // The relative-position window starts at (t_real - 1 - q) and is contiguous.
    let bd_base = row * cfg.p_stride + (cfg.t_real - 1u - q);
    for (var j = lid.x; j < cfg.t_real; j = j + 256u) {
        cache[j] = cfg.scale * (Content[base + j] + Bd[bd_base + j]);
    }
    workgroupBarrier();

    // The most negative f32, written as a decimal that Tint accepts: it rejects
    // any literal above f32::MAX instead of rounding it (3.4028235e38 lands on
    // the same f32 but is refused), and it folds bitcast<f32>(0xFF800000u) into
    // an -inf constant, which WGSL cannot spell. Indistinguishable from -inf for
    // a max accumulator over finite input.
    var m = -3.4028234e38;
    for (var j = lid.x; j < cfg.t_real; j = j + 256u) {
        if (cache[j] > m) { m = cache[j]; }
    }
    red_max[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }
        workgroupBarrier();
    }
    let row_max = red_max[0];
    workgroupBarrier();

    var acc = 0.0;
    for (var j = lid.x; j < cfg.t_real; j = j + 256u) {
        acc = acc + exp(cache[j] - row_max);
    }
    red_sum[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }
        workgroupBarrier();
    }
    let sum = red_sum[0];
    workgroupBarrier();

    for (var j = lid.x; j < tp; j = j + 256u) {
        if (j < cfg.t_real) {
            Out[base + j] = exp(cache[j] - row_max) / sum;
        } else {
            Out[base + j] = 0.0;
        }
    }
}
"#;

/// softmax rows of width up to 16384, f32 in/out.
pub(super) const P_SOFTMAX_F32: &str = r#"
struct Cfg { rows: u32, n: u32, stride: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
var<workgroup> red_max: array<f32, 256>;
var<workgroup> red_sum: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    if (row >= cfg.rows) { return; }
    let base = row * cfg.stride;
    // The most negative f32, written as a decimal that Tint accepts: it rejects
    // any literal above f32::MAX instead of rounding it (3.4028235e38 lands on
    // the same f32 but is refused), and it folds bitcast<f32>(0xFF800000u) into
    // an -inf constant, which WGSL cannot spell. Indistinguishable from -inf for
    // a max accumulator over finite input.
    var m = -3.4028234e38;
    for (var j = lid.x; j < cfg.n; j = j + 256u) {
        if (X[base + j] > m) { m = X[base + j]; }
    }
    red_max[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }
        workgroupBarrier();
    }
    let row_max = red_max[0];
    workgroupBarrier();
    var acc = 0.0;
    for (var j = lid.x; j < cfg.n; j = j + 256u) {
        acc = acc + exp(X[base + j] - row_max);
    }
    red_sum[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }
        workgroupBarrier();
    }
    let sum = red_sum[0];
    workgroupBarrier();
    for (var j = lid.x; j < cfg.stride; j = j + 256u) {
        Out[base + j] = select(0.0, exp(X[base + j] - row_max) / sum, j < cfg.n);
    }
}
"#;

/// heads (8, T, 128) f32 -> flat (T, 1024) f16.
pub(super) const P_H2F_F32: &str = r#"
struct Cfg { t: u32, words: u32, _s: u32, _o: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X:   array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= cfg.words) { return; }
    let t = w / 512u;
    let j = w % 512u;
    let h = j / 64u;
    let iw = (h * cfg.t + t) * 128u + (j % 64u) * 2u;
    Out[w] = pack2x16float(vec2<f32>(X[iw], X[iw + 1u]));
}
"#;

/// A[i] = A[i] + scale * B[i] — residual add with pre-add scaling (exact for scale=0.5).
pub(super) const P_ADD_SCALED: &str = r#"
struct Cfg { n2: u32, scale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> A: array<u32>;
@group(0) @binding(1) var<storage, read>       B: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n2) { return; }
    let av = unpack2x16float(A[i]);
    let bv = unpack2x16float(B[i]);
    A[i] = pack2x16float(vec2<f32>(av.x + cfg.scale * bv.x, av.y + cfg.scale * bv.y));
}
"#;

/// depthwise conv k=9 over time on (T, 1024); zero pad 4; w (1024, 9) f16.
pub(super) const P_DWCONV9: &str = r#"
struct Cfg { t: u32, words: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X: array<u32>;
@group(0) @binding(1) var<storage, read>       W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;   // word = (t, c pair) over (T, 1024) f16
    if (w >= cfg.words) { return; }
    let c = (w % 512u) * 2u;
    let t = w / 512u;
    var acc0 = 0.0; var acc1 = 0.0;
    for (var i = 0u; i < 9u; i = i + 1u) {
        var tt = t + i;
        if (tt < 4u) { continue; }
        tt = tt - 4u;
        if (tt >= cfg.t) { continue; }
        let xw = tt * 512u + (w % 512u);
        let xv = unpack2x16float(X[xw]);
        let w0 = c * 9u + i;
        let w1 = (c + 1u) * 9u + i;
        let wv0 = select(unpack2x16float(W[w0 / 2u]).x, unpack2x16float(W[w0 / 2u]).y, (w0 & 1u) == 1u);
        let wv1 = select(unpack2x16float(W[w1 / 2u]).x, unpack2x16float(W[w1 / 2u]).y, (w1 & 1u) == 1u);
        acc0 = acc0 + xv.x * wv0;
        acc1 = acc1 + xv.y * wv1;
    }
    Out[w] = pack2x16float(vec2<f32>(acc0, acc1));
}
"#;

/// BatchNorm(inference) + SiLU on (T, 1024): out = silu(x*a + b), a/b f32 per channel.
pub(super) const P_BN_SILU: &str = r#"
struct Cfg { t: u32, words: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       X: array<u32>;
@group(0) @binding(1) var<storage, read>       AB: array<f32>;   // a[1024] then b[1024]
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;
const LOG2E: f32 = 1.4426950408889634;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= cfg.words) { return; }
    let c = (w % 512u) * 2u;
    let xv = unpack2x16float(X[w]);
    let y0 = xv.x * AB[c] + AB[1024u + c];
    let y1 = xv.y * AB[c + 1u] + AB[1024u + c + 1u];
    let s0 = y0 / (1.0 + exp2(-y0 * LOG2E));
    let s1 = y1 / (1.0 + exp2(-y1 * LOG2E));
    Out[w] = pack2x16float(vec2<f32>(s0, s1));
}
"#;
