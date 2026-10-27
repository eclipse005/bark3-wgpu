//! WGSL compute kernels for the Bark 3.0 Pro wgpu port.
//! Generic kernel set adapted from moss-transcribe-diarize-wgpu / qwen3-asr-wgpu
//! (Apache-2.0): f16-storage / f32-compute, one f16 rounding per output.
#![allow(dead_code)]
pub const HALF_AT: &str = r#"
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
"#;

/// Bit-exact software `expf` (Cody-Waite reduction + integer-scaled 2^q).
/// Verbatim from the qwen3-asr-wgpu kernel set; the WGSL builtin `exp` is not
/// reliable enough for the GQA decode path.
pub const EXP_BT: &str = r#"
fn lead32(v0: u32) -> i32 {
    var n = 0i;
    var v = v0;
    if ((v & 0xFFFF0000u) != 0u) { n = n + 16; v = v >> 16u; }
    if ((v & 0xFF00u) != 0u)     { n = n + 8;  v = v >> 8u;  }
    if ((v & 0xF0u) != 0u)       { n = n + 4;  v = v >> 4u;  }
    if ((v & 0xCu) != 0u)        { n = n + 2;  v = v >> 2u;  }
    if ((v & 0x2u) != 0u)        { n = n + 1; }
    return n;
}
fn shr64(hi0: u32, lo0: u32, t0: i32) -> vec3<u32> {
    if (t0 <= 0) { return vec3<u32>(hi0, lo0, 0u); }
    if (t0 >= 64) { return vec3<u32>(0u, 0u, select(0u, 1u, (hi0 | lo0) != 0u)); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        let lost = select(0u, 1u, lo0 != 0u) | select(0u, 1u, (hi0 & ((1u << s) - 1u)) != 0u);
        return vec3<u32>(0u, hi0 >> s, lost);
    }
    let s = u32(t0);
    let lost = select(0u, 1u, (lo0 & ((1u << s) - 1u)) != 0u);
    return vec3<u32>(hi0 >> s, (lo0 >> s) | (hi0 << (32u - s)), lost);
}
fn shl64(hi0: u32, lo0: u32, t0: i32) -> vec2<u32> {
    if (t0 <= 0) { return vec2<u32>(hi0, lo0); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        return vec2<u32>(lo0 << s, 0u);
    }
    let s = u32(t0);
    return vec2<u32>((hi0 << s) | (lo0 >> (32u - s)), lo0 << s);
}
fn bit64(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 >= 32) { return (hi >> u32(pos0 - 32)) & 1u; }
    return (lo >> u32(pos0)) & 1u;
}
fn stickyBelow(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 <= 0) { return 0u; }
    if (pos0 >= 64) { return select(0u, 1u, (hi | lo) != 0u); }
    if (pos0 >= 32) {
        let s = u32(pos0) - 32u;
        return select(0u, 1u, lo != 0u) | select(0u, 1u, (hi & ((1u << s) - 1u)) != 0u);
    }
    let s = u32(pos0);
    return select(0u, 1u, (lo & ((1u << s) - 1u)) != 0u);
}
fn fma_int(a: f32, b: f32, c: f32) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let bc = bitcast<u32>(c);
    let ea0 = i32((ba >> 23u) & 0xFFu);
    let eb0 = i32((bb >> 23u) & 0xFFu);
    let ec0 = i32((bc >> 23u) & 0xFFu);
    if (ea0 == 255 || eb0 == 255 || ec0 == 255) { return a * b + c; }
    var ma = ba & 0x7FFFFFu;
    var mb = bb & 0x7FFFFFu;
    var mc = bc & 0x7FFFFFu;
    var ea = ea0; var eb = eb0; var ec = ec0;
    if (ea0 == 0) { ea = 1; } else { ma = ma | 0x800000u; }
    if (eb0 == 0) { eb = 1; } else { mb = mb | 0x800000u; }
    if (ec0 == 0) { ec = 1; } else { mc = mc | 0x800000u; }
    let sa = ba >> 31u;
    let sbS = bb >> 31u;
    let scS = bc >> 31u;
    if (ma == 0u || mb == 0u) { return c; }
    let a1 = ma >> 12u; let a0 = ma & 0xFFFu;
    let b1 = mb >> 12u; let b0 = mb & 0xFFFu;
    let t0 = a0 * b0;
    let t1 = a0 * b1 + a1 * b0;
    let t2 = a1 * b1;
    let c1w = (t0 >> 12u) + t1;
    let w0 = t0 & 0xFFFu;
    let w1 = c1w & 0xFFFu;
    let w2 = (c1w >> 12u) + t2;
    var pHi = w2 >> 8u;
    var pLo = ((w2 & 0xFFu) << 24u) | (w1 << 12u) | w0;
    var ep = ea + eb - 300;
    if (pHi < 0x8000u) {
        pHi = (pHi << 1u) | (pLo >> 31u);
        pLo = pLo << 1u;
        ep = ep - 1;
    }
    let EP = ep + 47;
    var E = EP + 4;
    var hasC = (mc != 0u);
    if (hasC) { E = max(E, ec - 127 + 4); }
    let shP = E - EP;
    var sticky = 0u;
    var Mp = shl64(pHi, pLo, 16 - shP);
    if (shP > 16) {
        let r = shr64(pHi, pLo, shP - 16);
        Mp = vec2<u32>(r.x, r.y);
        sticky = r.z;
    }
    var McHi = 0u; var McLo = 0u;
    if (hasC) {
        let shC = E - (ec - 127);
        if (shC <= 40) {
            let r = shl64(0u, mc, 40 - shC);
            McHi = r.x; McLo = r.y;
        } else {
            let r = shr64(0u, mc, shC - 40);
            McHi = r.x; McLo = r.y;
            sticky = sticky | r.z;
        }
    }
    let spSign = sa ^ sbS;
    var rHi = 0u; var rLo = 0u; var rSign = 0u;
    if (!hasC) {
        rHi = Mp.x; rLo = Mp.y; rSign = spSign;
    } else if (spSign == scS) {
        let lo = Mp.y + McLo;
        let carry = select(0u, 1u, lo < Mp.y);
        rHi = Mp.x + McHi + carry;
        rLo = lo;
        rSign = spSign;
    } else {
        var gt = (Mp.x > McHi) || (Mp.x == McHi && Mp.y > McLo);
        var eq = (Mp.x == McHi && Mp.y == McLo);
        if (eq) { return bitcast<f32>(0u); }
        var bigHi = Mp.x; var bigLo = Mp.y; var smlHi = McHi; var smlLo = McLo;
        rSign = spSign;
        if (!gt) {
            bigHi = McHi; bigLo = McLo; smlHi = Mp.x; smlLo = Mp.y;
            rSign = scS;
        }
        let borrow = select(0u, 1u, bigLo < smlLo);
        rHi = bigHi - smlHi - borrow;
        rLo = bigLo - smlLo;
    }
    if (rHi == 0u && rLo == 0u) { return bitcast<f32>(0u); }
    var lb: i32;
    if (rHi != 0u) { lb = lead32(rHi) + 32; } else { lb = lead32(rLo); }
    let F = E - 63;
    var eUnb = F + lb;
    let signBit = rSign << 31u;
    if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
    if (eUnb >= -126) {
        var mR: u32;
        let R = lb - 23;
        if (R <= 0) {
            let r = shl64(rHi, rLo, -R);
            mR = r.y;
        } else {
            if (R >= 32) { mR = rHi >> u32(R - 32); }
            else { mR = (rHi << u32(32 - R)) | (rLo >> u32(R)); }
            var g = 0u; var rr = 0u; var st = sticky;
            if (R >= 2) {
                g = bit64(rHi, rLo, R - 1);
                rr = bit64(rHi, rLo, R - 2);
                st = st | stickyBelow(rHi, rLo, R - 2);
            } else {
                g = bit64(rHi, rLo, 0);
            }
            if (g != 0u && (rr != 0u || st != 0u || (mR & 1u) == 1u)) {
                mR = mR + 1u;
                if (mR == 0x1000000u) { mR = 0x800000u; eUnb = eUnb + 1; }
                if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
            }
        }
        return bitcast<f32>(signBit | (u32(eUnb + 127) << 23u) | (mR & 0x7FFFFFu));
    }
    let shiftS = F + 149;
    var k: u32;
    if (shiftS >= 0) {
        let r = shl64(rHi, rLo, shiftS);
        k = r.y;
    } else {
        let t = -shiftS;
        if (t >= 64) { return bitcast<f32>(signBit); }
        let r = shr64(rHi, rLo, t);
        k = r.y;
        var g = 0u; var rr = 0u; var st = sticky;
        if (t >= 2) {
            g = bit64(rHi, rLo, t - 1);
            rr = bit64(rHi, rLo, t - 2);
            st = st | stickyBelow(rHi, rLo, t - 2);
        } else {
            g = bit64(rHi, rLo, 0);
        }
        if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) {
            k = k + 1u;
            if (k == 0x800000u) { return bitcast<f32>(signBit | 0x800000u); }
        }
    }
    if (k == 0u) { return bitcast<f32>(signBit); }
    return bitcast<f32>(signBit | k);
}
fn scale_pow2(Abits: u32, q: u32) -> u32 {
    let sgn = Abits & 0x80000000u;
    if ((Abits & 0x7FFFFFFFu) == 0u) { return sgn; }
    let mA = (Abits & 0x7FFFFFu) | 0x800000u;
    let eA = i32((Abits >> 23u) & 0xFFu) - 127;
    let eRes = eA + i32(q) - 126;
    if (eRes >= 128) { return sgn | 0x7F800000u; }
    if (eRes >= -126) {
        return sgn | (u32(eRes + 127) << 23u) | (Abits & 0x7FFFFFu);
    }
    let s = -126 - eRes;
    if (s >= 25) { return sgn; }
    let k = mA >> u32(s);
    var kR = k;
    var g = 0u; var rr = 0u; var st = 0u;
    if (s >= 2) {
        g = (mA >> u32(s - 1)) & 1u;
        rr = (mA >> u32(s - 2)) & 1u;
        st = select(0u, 1u, (mA & ((1u << u32(s - 2)) - 1u)) != 0u);
    } else {
        g = (mA >> u32(s - 1)) & 1u;
    }
    if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) { kR = k + 1u; }
    if (kR >= 0x800000u) { return sgn | 0x800000u; }
    return sgn | kR;
}
fn expf_bt(x: f32) -> f32 {
    let f5 = clamp(fma_int(x, bitcast<f32>(0x3BBB989Du), 0.5), 0.0, 1.0);
    let b5 = bitcast<u32>(f5);
    var f8 = 12582913u;
    if ((b5 & 0x7FFFFFFFu) != 0u) {
        let raw_e = i32((b5 >> 23u) & 0xFFu) - 127;
        let m = select(b5 & 0x7FFFFFu, (b5 & 0x7FFFFFu) | 0x800000u, raw_e != -127);
        let ee = select(-126, raw_e, raw_e != -127);
        let pp = (m << 8u) - (m << 2u);
        let shift = u32(23 - ee);
        let q = select(0u, pp >> shift, shift < 32u);
        f8 = 12582913u + q;
    }
    let f8f = bitcast<f32>(0x4B000000u | (f8 - 0x800000u));
    let f10 = -(f8f + bitcast<f32>(0xCB40007Fu));
    let f12 = fma_int(x, bitcast<f32>(0x3FB8AA3Bu), f10);
    let f14 = fma_int(x, bitcast<f32>(0x32A57060u), f12);
    let q = f8 - 12582913u;
    return bitcast<f32>(scale_pow2(bitcast<u32>(exp2(f14)), q));
}
"#;

// ═══════════════════════════ rms_norm ═══════════════════════════
// One workgroup per row. 256 threads × 2 words/thread; the in-thread fold
// reproduces CUDA's `block_for_reduction(1024)` 1024-lane tree (the CUDA
// kernel's upper 512 lanes are idle zero-lanes, so the effective tree is the
// 512-word tree with pairing (w, w+256), (w, w+128), …).
// out[j] = x[j] * rsqrt(mean(x^2) + eps) * w[j].
pub const RMS_NORM: &str = r#"
struct Cfg { eps: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read>       Wt:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;

const LAST: u32 = 1024u;
const LAST2: u32 = 512u;
const BS: u32 = 256u;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x * LAST2;
    let w0 = lid.x;
    let w1 = lid.x + 256u;
    let v0 = unpack2x16float(X[row + w0]);
    let v1 = unpack2x16float(X[row + w1]);
    var local = (v0.x * v0.x + v0.y * v0.y) + (v1.x * v1.x + v1.y * v1.y);
    red[lid.x] = local;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
        workgroupBarrier();
    }
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + cfg.eps);
    workgroupBarrier();
    let xv0 = unpack2x16float(X[row + w0]);
    let wv0 = unpack2x16float(Wt[w0]);
    let xv1 = unpack2x16float(X[row + w1]);
    let wv1 = unpack2x16float(Wt[w1]);
    Out[row + w0] = pack2x16float(vec2<f32>(xv0.x * inv_rms * wv0.x, xv0.y * inv_rms * wv0.y));
    Out[row + w1] = pack2x16float(vec2<f32>(xv1.x * inv_rms * wv1.x, xv1.y * inv_rms * wv1.y));
}
"#;

// ═══════════════════════════ layer_norm (with bias) ═══════════════════════════
// One workgroup per row. 256 threads × 4 virtual element-lanes; the in-thread
// fold reproduces CUDA's 1024-lane tree: level 1 pairs (t, t+512), level 2
// (t, t+256) — i.e. (k0+k2) then (k0+k1) — followed by the 256-thread tree.
// Single-pass biased variance, as CUDA: var = E[x^2] - E[x]^2.
pub const LAYER_NORM: &str = r#"
struct LnCfg { last: u32, outer: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read>       Wgt: array<u32>;
@group(0) @binding(2) var<storage, read>       Bia: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<u32>;

const LAST: u32 = 1024u;
const LAST2: u32 = 512u;
const BP: u32 = 256u;
const BT: u32 = 1024u;

var<workgroup> red_s: array<f32, 256>;
var<workgroup> red_q: array<f32, 256>;

@compute @workgroup_size(256)
fn layernorm(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    if (row >= cfg.outer) { return; }
    let base = row * LAST;
    // four virtual element lanes per thread: v = lid + k*256
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
    var q0 = 0.0; var q1 = 0.0; var q2 = 0.0; var q3 = 0.0;
    let e0 = half_at(unpack2x16float(Src[(base + lid.x) >> 1u]), base + lid.x);
    let e1 = half_at(unpack2x16float(Src[(base + lid.x + 256u) >> 1u]), base + lid.x + 256u);
    let e2 = half_at(unpack2x16float(Src[(base + lid.x + 512u) >> 1u]), base + lid.x + 512u);
    let e3 = half_at(unpack2x16float(Src[(base + lid.x + 768u) >> 1u]), base + lid.x + 768u);
    s0 = e0; s1 = e1; s2 = e2; s3 = e3;
    q0 = e0 * e0; q1 = e1 * e1; q2 = e2 * e2; q3 = e3 * e3;
    // fold level 1: (k0,k2), (k1,k3)
    let sA = s0 + s2; let sB = s1 + s3;
    let qA = q0 + q2; let qB = q1 + q3;
    // fold level 2: (k0,k1)
    let sL = sA + sB;
    let qL = qA + qB;
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
    let mean = red_s[0] / f32(LAST);
    let var_ = red_q[0] / f32(LAST) - mean * mean;
    let inv_std = inverseSqrt(var_ + cfg.eps);
    workgroupBarrier();
    for (var w = lid.x; w < LAST2; w = w + BP) {
        let sv = unpack2x16float(Src[base / 2u + w]);
        let wv = unpack2x16float(Wgt[w]);
        let bv = unpack2x16float(Bia[w]);
        Dst[base / 2u + w] = pack2x16float(vec2<f32>(
            (sv.x - mean) * inv_std * wv.x + bv.x,
            (sv.y - mean) * inv_std * wv.y + bv.y));
    }
}
"#;

// ═══════════════════════════ add_bias_inplace ═══════════════════════════
// x[outer, last] += bias[last] (broadcast over rows).
pub const ADD_BIAS: &str = r#"
struct BiasCfg { outer: u32, last: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<u32>;
@group(0) @binding(1) var<storage, read>       B: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: BiasCfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total2 = cfg.outer * (cfg.last / 2u);
    if (i >= total2) { return; }
    let j2 = i % (cfg.last / 2u);
    let xv = unpack2x16float(X[i]);
    let bv = unpack2x16float(B[j2]);
    X[i] = pack2x16float(vec2<f32>(xv.x + bv.x, xv.y + bv.y));
}
"#;

// ═══════════════════════════ add_inplace (flat) ═══════════════════════════
pub const ADD: &str = r#"
struct NCfg { n2: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> A: array<u32>;
@group(0) @binding(1) var<storage, read>       B: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: NCfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n2) { return; }
    let av = unpack2x16float(A[i]);
    let bv = unpack2x16float(B[i]);
    A[i] = pack2x16float(vec2<f32>(av.x + bv.x, av.y + bv.y));
}
"#;

// ═══════════════════════════ silu_mul_split ═══════════════════════════
// gu[o, 0:inter] = gate, gu[o, inter:2*inter] = up → out[o,c] = silu(gate)*up.
// Uses exp2(-x*log2e) like the reference port (matches the approx-exp form).
pub const SILU_MUL_SPLIT: &str = r#"
struct Cfg { inter2: u32, total2: u32, gx: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       Gu:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const LOG2E: f32 = 1.4426950408889634;

@compute @workgroup_size(256)
fn silu_mul_split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * (cfg.gx * 256u);
    if (i >= cfg.total2) { return; }
    let row = i / cfg.inter2;
    let c = i - row * cfg.inter2;
    let base = row * cfg.inter2 * 2u;
    let gv = unpack2x16float(Gu[base + c]);
    let uv = unpack2x16float(Gu[base + cfg.inter2 + c]);
    let sx = 1.0 / (1.0 + exp2(-gv.x * LOG2E));
    let sy = 1.0 / (1.0 + exp2(-gv.y * LOG2E));
    Out[i] = pack2x16float(vec2<f32>(gv.x * sx * uv.x, gv.y * sy * uv.y));
}
"#;

// ═══════════════════════════ silu_inplace (VQ adaptor) ═══════════════════════════
pub const SILU_INPLACE: &str = r#"
struct NCfg { n2: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<u32>;
@group(0) @binding(1) var<uniform>             cfg: NCfg;

const LOG2E: f32 = 1.4426950408889634;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n2) { return; }
    let v = unpack2x16float(X[i]);
    let sx = 1.0 / (1.0 + exp2(-v.x * LOG2E));
    let sy = 1.0 / (1.0 + exp2(-v.y * LOG2E));
    X[i] = pack2x16float(vec2<f32>(v.x * sx, v.y * sy));
}
"#;

// ═══════════════════════════ gelu_inplace (encoder FFN) ═══════════════════════════
// Exact GELU with the A&S 7.1.26 erf (max abs err 1.5e-7, far below f16).
pub const GELU: &str = r#"
struct NCfg { n2: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<u32>;
@group(0) @binding(1) var<uniform>             cfg: NCfg;

const FRAC_1_SQRT_2: f32 = 0.70710678118654752440;

fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n2) { return; }
    let v = unpack2x16float(X[i]);
    X[i] = pack2x16float(vec2<f32>(
        0.5 * v.x * (1.0 + erf_approx(v.x * FRAC_1_SQRT_2)),
        0.5 * v.y * (1.0 + erf_approx(v.y * FRAC_1_SQRT_2))));
}
"#;

// ═══════════════════════════ softmax (prefill / encoder attention) ═══════════════════════════
// One workgroup per score row; `bt` virtual lanes (matching CUDA's
// `block_for_reduction(n)` tree width exactly) emulated on min(256, bt)
// physical threads. causal: valid = min(n, row+1); else valid = n.
// Columns >= valid write 0.
pub fn softmax(bt: usize) -> String {
    assert!(bt.is_power_of_two() && (32..=1024).contains(&bt));
    let threads = bt.min(256);
    let r = bt / threads;
    // lane-k loops for the max pass and the sum pass
    let mut m_loads = String::new();
    let mut s_loads = String::new();
    for k in 0..r {
        m_loads.push_str(&format!(
            "    for (var j = lid.x + {k}u * BP; j < valid; j = j + BT) {{\n\
             \x20 let v = half_at(unpack2x16float(X[base_x + (j >> 1u)]), j) * cfg.scale;\n\
             \x20 if (v > m{k}) {{ m{k} = v; }}\n    }}\n"
        ));
        s_loads.push_str(&format!(
            "    for (var j = lid.x + {k}u * BP; j < valid; j = j + BT) {{\n\
             \x20 s{k} = s{k} + exp(half_at(unpack2x16float(X[base_x + (j >> 1u)]), j) * cfg.scale - row_max);\n\
             \x20 }}\n"
        ));
    }
    // fold the first log2(r) tree levels in thread-local order: level pairs
    // (k, k + half) for k < half, survivors keep names m0..m{half-1}.
    let mut m_folds = String::new();
    let mut s_folds = String::new();
    let mut half = r / 2;
    while half >= 1 {
        for k in 0..half {
            m_folds.push_str(&format!(
                "    let fmx_{half}_{k} = max(m{k}, m{});\n",
                k + half
            ));
            s_folds.push_str(&format!(
                "    let fsx_{half}_{k} = (s{k} + s{});\n",
                k + half
            ));
        }
        for k in 0..half {
            m_folds.push_str(&format!("    m{k} = fmx_{half}_{k};\n"));
            s_folds.push_str(&format!("    s{k} = fsx_{half}_{k};\n"));
        }
        half /= 2;
    }
    let mut mdecl = String::new();
    for k in 0..r {
        mdecl.push_str(&format!("    var m{k} = bitcast<f32>(0xFF800000u);\n"));
    }
    let mut sdecl = String::new();
    for k in 0..r {
        sdecl.push_str(&format!("    var s{k} = 0.0;\n"));
    }
    format!(
        "struct Cfg {{ n_w: u32, n_x: u32, n_valid: u32, m: u32, head_stride: u32, scale: f32, causal: u32, row0: u32 }};\n\
@group(0) @binding(0) var<storage, read>       X:   array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;\n\
@group(0) @binding(2) var<uniform>             cfg: Cfg;\n\
const BP: u32 = {threads}u;\n\
const BT: u32 = {bt}u;\n\
var<workgroup> red_max: array<f32, {threads}>;\n\
var<workgroup> red_sum: array<f32, {threads}>;\n\
@compute @workgroup_size({threads})\n\
fn softmax(@builtin(workgroup_id) wgid: vec3<u32>,\n\
           @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
    let head = wgid.x / cfg.m;\n\
    let pos = wgid.x % cfg.m;\n\
    let row = head * cfg.head_stride + pos;\n\
    let base_x = row * cfg.n_x;\n\
    let base_o = row * cfg.n_w;\n\
    let valid = select(cfg.n_valid, min(cfg.n_valid, cfg.row0 + pos + 1u), cfg.causal != 0u);\n\
    if (valid == 0u) {{\n\
        for (var w = lid.x; w < cfg.n_w; w = w + BP) {{ Out[base_o + w] = 0u; }}\n\
        return;\n\
    }}\n\
{mdecl}{m_loads}{m_folds}\
    red_max[lid.x] = m0;\n\
    workgroupBarrier();\n\
    for (var s = BP >> 1u; s > 0u; s = s >> 1u) {{\n\
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}\n\
        workgroupBarrier();\n\
    }}\n\
    let row_max = red_max[0];\n\
    workgroupBarrier();\n\
{sdecl}{s_loads}{s_folds}\
    red_sum[lid.x] = s0;\n\
    workgroupBarrier();\n\
    for (var s = BP >> 1u; s > 0u; s = s >> 1u) {{\n\
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}\n\
        workgroupBarrier();\n\
    }}\n\
    let inv_sum = 1.0 / red_sum[0];\n\
    workgroupBarrier();\n\
    for (var w = lid.x; w < cfg.n_w; w = w + BP) {{\n\
        let j0 = w * 2u;\n\
        let j1 = j0 + 1u;\n\
        var e0 = 0.0;\n\
        var e1 = 0.0;\n\
        if (j0 < valid) {{\n\
            e0 = exp(half_at(unpack2x16float(X[base_x + w]), j0) * cfg.scale - row_max) * inv_sum;\n\
        }}\n\
        if (j1 < valid) {{\n\
            e1 = exp(half_at(unpack2x16float(X[base_x + w]), j1) * cfg.scale - row_max) * inv_sum;\n\
        }}\n\
        Out[base_o + w] = pack2x16float(vec2<f32>(e0, e1));\n\
    }}\n\
}}\n",
        threads = threads,
        bt = bt,
        mdecl = mdecl,
        sdecl = sdecl,
        m_loads = m_loads,
        s_loads = s_loads,
        m_folds = m_folds,
        s_folds = s_folds,
    )
}

// ═══════════════════════════ qkv_extract (decoder) ═══════════════════════════
// Per (position, head) workgroup: per-head RMSNorm + rotate-half RoPE for Q/K,
// raw copy for V, K/V written into the per-layer cache at row `start+is`.
// Mirrors CUDA `qkv_extract_qkv_norm_rotary_cache_f16`.

pub fn format_f32(v: f32) -> String {
    let bits = v.to_bits();
    let sign = if bits >> 31 == 1 { "-" } else { "" };
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;
    if exp == 0 && man == 0 {
        return format!("{sign}0.0");
    }
    let man = man << 1;
    let e = exp - 127;
    format!("{sign}0x1.{man:06x}p{e}")
}

pub const PREFILL_GEMM_TM: usize = 8;
pub const PREFILL_GEMM_TN: usize = 8;
pub const PREFILL_GEMM_BK: usize = 16;
pub const PREFILL_GEMM_BM: usize = 16 * PREFILL_GEMM_TM;
pub const PREFILL_GEMM_BN: usize = 16 * PREFILL_GEMM_TN;
const GEMM_UNROLL: usize = 4;

pub fn prefill_gemm(
    transb: bool,
    beta: bool,
    bias: bool,
    causal_skip: bool,
    causal_k: bool,
) -> String {
    let tm = PREFILL_GEMM_TM;
    let tn = PREFILL_GEMM_TN;
    let bk = PREFILL_GEMM_BK;
    let bm = PREFILL_GEMM_BM;
    let bn = PREFILL_GEMM_BN;
    let aq = bm / 4;
    let bq = bn / 4;
    let cstep = bm * bk / 4 / 2;

    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<u32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    if bias {
        s.push_str("@group(0) @binding(4) var<storage, read> Bias: array<u32>;\n");
    }
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst CSTEP: u32 = {cstep}u;\n\
         const TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const TRANSB: u32 = {}u;\nconst BETA: u32 = {}u;\nconst BIAS: u32 = {}u;\n\
         const CAUSAL: u32 = {}u;\nconst CAUSAL_K: u32 = {}u;\n",
        u32::from(transb),
        u32::from(beta),
        u32::from(bias),
        u32::from(causal_skip),
        u32::from(causal_k),
    ));
    s.push_str(&format!(
        "var<workgroup> A4: array<vec4<f32>, {}>;\n",
        bm * bk / 4
    ));
    s.push_str(&format!(
        "var<workgroup> B4: array<vec4<f32>, {}>;\n",
        bn * bk / 4
    ));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n\
         \x20 let p = unpack2x16float(w);\n\
         \x20 return select(p.x, p.y, odd);\n\
         }\n",
    );
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda / 2u;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         if (CAUSAL == 1u && n0 > gd.row0 + m0 + BM - 1u) { return; }\n\
         let klim = select(gd.k, min(gd.k, gd.row0 + m0 + BM), CAUSAL_K == 1u);\n\
         let c0 = ty * 16u + tx;\n\
         \x20 let crg = c0 % AQ;\n let ck = c0 / AQ;\n\
         \x20 let cvodd = ((ck & 1u) == 1u);\n\
         \x20 let bop = c0 % BQ;\n\
         \x20 let bog = ((bop >> 4u) << 3u) | (bop & 7u);\n\
         \x20 let bos = (bop >> 3u) & 1u;\n",
    );
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }
    // initial A tile loads (k = 0..15): two chunks of 4 rows at column ck / ck+8
    for half in 0..2 {
        let kw = format!("(0u + ck + {}u)", half * 8);
        let mut comps = String::new();
        for e in 0..4 {
            comps.push_str(&format!(
                "\x20 halve(A[abase + (m0 + 4u * crg + {e}u) * alda + {kw} / 2u], cvodd),"
            ));
        }
        s.push_str(&format!(
            "  A4[c0{}] = vec4<f32>({comps});\n",
            if half == 0 {
                String::new()
            } else {
                " + CSTEP".into()
            }
        ));
    }
    // initial B tile loads
    for half in 0..2 {
        if transb {
            let base = format!(
                "wb + (0u + ck{}) * (gd.n / 2u)",
                if half == 0 {
                    String::new()
                } else {
                    " + 8u".into()
                }
            );
            s.push_str(&format!(
                "  B4[c0{cs}] = vec4<f32>(halve(W[{base} + {wo}], false), halve(W[{base} + {wo}], true), halve(W[{base} + {wo} + 1u], false), halve(W[{base} + {wo} + 1u], true));\n",
                cs = if half == 0 { String::new() } else { " + CSTEP".into() },
                wo = "(n0 + 8u * bog + 4u * bos) / 2u"
            ));
        } else {
            let kw = format!("(0u + ck + {}u)", half * 8);
            let mut comps = String::new();
            for e in 0..4 {
                comps.push_str(&format!(
                    "\x20 halve(W[wb + (n0 + 8u * bog + 4u * bos + {e}u) * kk + {kw} / 2u], cvodd),"
                ));
            }
            s.push_str(&format!(
                "  B4[c0{}] = vec4<f32>({comps});\n",
                if half == 0 {
                    String::new()
                } else {
                    " + CSTEP".into()
                }
            ));
        }
    }
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= klim) { break; }\n  let kn = k0 + BK;\n");
    for e in 0..8 {
        s.push_str(&format!("  var pfa{e}: u32 = 0u;\n"));
    }
    for e in 0..8 {
        s.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
    }
    s.push_str("  if (kn < klim) {\n");
    for half in 0..2 {
        let kw = format!("(kn + ck + {}u)", half * 8);
        for e in 0..4 {
            s.push_str(&format!(
                "   pfa{} = A[abase + (m0 + 4u * crg + {e}u) * alda + {kw} / 2u];\n",
                half * 4 + e
            ));
        }
    }
    for half in 0..2 {
        if transb {
            let base = format!(
                "wb + (kn + ck{}) * (gd.n / 2u)",
                if half == 0 {
                    String::new()
                } else {
                    " + 8u".into()
                }
            );
            s.push_str(&format!(
                "   pfb{} = W[{base} + {wo}];\n   pfb{} = W[{base} + {wo} + 1u];\n",
                half * 2,
                half * 2 + 1,
                wo = "(n0 + 8u * bog + 4u * bos) / 2u"
            ));
        } else {
            let kw = format!("(kn + ck + {}u)", half * 8);
            for e in 0..4 {
                s.push_str(&format!(
                    "   pfb{} = W[wb + (n0 + 8u * bog + 4u * bos + {e}u) * kk + {kw} / 2u];\n",
                    half * 4 + e
                ));
            }
        }
    }
    s.push_str("  }\n");
    // compute: tile-internal q-loop, unroll 4
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    for u in 0..GEMM_UNROLL {
        let idx = format!("(q0 + {u}u)");
        s.push_str(&format!("   let av_{u} = A4[{idx} * AQ + ty * 2u];\n"));
        s.push_str(&format!("   let aw_{u} = A4[{idx} * AQ + ty * 2u + 1u];\n"));
        s.push_str(&format!(
            "   let bv_{u} = B4[{idx} * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n"
        ));
        s.push_str(&format!(
            "   let bw_{u} = B4[{idx} * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
        ));
        for i in 0..tm {
            let a = if i < 4 {
                format!("av_{u}[{i}]")
            } else {
                format!("aw_{u}[{}]", i - 4)
            };
            for j in 0..tn {
                let b = if j < 4 {
                    format!("bv_{u}[{j}]")
                } else {
                    format!("bw_{u}[{}]", j - 4)
                };
                s.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
            }
        }
    }
    s.push_str(&format!("   q0 = q0 + {GEMM_UNROLL}u;\n  }}\n"));
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < klim) {\n");
    for half in 0..2 {
        let mut comps = String::new();
        for e in 0..4 {
            comps.push_str(&format!("\x20 halve(pfa{}, cvodd),", half * 4 + e));
        }
        s.push_str(&format!(
            "   A4[c0{}] = vec4<f32>({comps});\n",
            if half == 0 {
                String::new()
            } else {
                " + CSTEP".into()
            }
        ));
    }
    for half in 0..2 {
        if transb {
            s.push_str(&format!(
                "   B4[c0{cs}] = vec4<f32>(halve(pfb{r}, false), halve(pfb{r}, true), halve(pfb{r2}, false), halve(pfb{r2}, true));\n",
                cs = if half == 0 { String::new() } else { " + CSTEP".into() },
                r = half * 2,
                r2 = half * 2 + 1
            ));
        } else {
            let mut comps = String::new();
            for e in 0..4 {
                comps.push_str(&format!("\x20 halve(pfb{}, cvodd),", half * 4 + e));
            }
            s.push_str(&format!(
                "   B4[c0{}] = vec4<f32>({comps});\n",
                if half == 0 {
                    String::new()
                } else {
                    " + CSTEP".into()
                }
            ));
        }
    }
    s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");

    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for e in 0..tn / 2 {
            let je = 2 * e;
            let jo = je + 1;
            s.push_str(&format!(
                "let we{i}_{e} = (cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {je}u) / 2u;\n"
            ));
            s.push_str(&format!("var v{i}{e} = vec2<f32>(c{i}{je}, c{i}{jo});\n"));
            if bias {
                s.push_str(&format!(
                    "v{i}{e} = v{i}{e} + unpack2x16float(Bias[(n0 + tx * {tn}u + {je}u) / 2u]);\n"
                ));
            }
            s.push_str(&format!(
                "if (BETA == 1u) {{\n  let old = unpack2x16float(C[we{i}_{e}]);\n  v{i}{e} = v{i}{e} + old;\n}}\n"
            ));
            s.push_str(&format!("C[we{i}_{e}] = pack2x16float(v{i}{e});\n"));
        }
    }
    s.push_str("}\n");
    s
}
