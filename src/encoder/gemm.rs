//! WGSL generators for the mixed-precision GEMM (A f32 · W f16 · C f32)
//! used by the attention score path.
//!
//! The kernel text is assembled at runtime from tile parameters, so the
//! emitted string must stay byte-identical to what naga saw before.

// ───────────────── mixed GEMM: A f32 · W f16 · C f32 (score-path precision) ─────────────────
// Same 128x128 tile / 16x16 threads / 8x8 per-thread / BK=16 double-buffer structure as the
// f16 GEMM (see shaders.rs), but the A operand and C output are f32 words: the attention
// score path (content scores, bd, attn probs) keeps full f32 through softmax.

pub fn gemm_mix(transb: bool) -> String {
    gemm_mix_param(transb, 16)
}

fn gemm_mix_param(transb: bool, bk: usize) -> String {
    let (tm, tn, bm, bn) = (8usize, 8, 128, 128);
    let halves = bk / 8;
    // Padded row-group stride: makes the coalesced tile stores bank-conflict free.
    let (aq, bq) = (bm / 4 + 1, bn / 4 + 1);
    let cstep = 8 * bq;
    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst CSTEP: u32 = {cstep}u;\n\
         const TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\nconst TRANSB: u32 = {}u;\n",
        u32::from(transb)
    ));
    s.push_str(&format!(
        "var<workgroup> A4: array<vec4<f32>, {}>;\nvar<workgroup> B4: array<vec4<f32>, {}>;\n",
        bk * aq,
        bk * bq
    ));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n let p = unpack2x16float(w);\n return select(p.x, p.y, odd);\n }\n",
    );
    // Thread -> tile element mapping.  A warp now spans 8 consecutive k for a few
    // row groups instead of 32 different rows at one k, so global loads coalesce.
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda;              // f32 words per A row\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         let c0 = ty * 16u + tx;\n\
         let arg = c0 / 8u;\n let akl = c0 % 8u;\n\
         let bkl = c0 % 8u;\n let bop = c0 / 8u;\n\
         let bog = ((bop >> 4u) << 3u) | (bop & 7u);\n\
         let bos = (bop >> 3u) & 1u;\n\
         let bodd = ((bkl & 1u) == 1u);\n\
         let tbop = c0 % 32u;\n let tck = c0 / 32u;\n\
         let tbog = ((tbop >> 4u) << 3u) | (tbop & 7u);\n\
         let tbos = (tbop >> 3u) & 1u;\n",
    );
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }
    // initial A tile: element = 4 rows of one k column
    for h in 0..halves {
        let mut comps = String::new();
        for e in 0..4 {
            comps.push_str(&format!(
                " A[abase + (m0 + 4u * arg + {e}u) * alda + ({}u + akl)],",
                h * 8
            ));
        }
        s.push_str(&format!(
            "  A4[({}u + akl) * AQ + arg] = vec4<f32>({comps});\n",
            h * 8
        ));
    }
    // initial B tile
    for h in 0..halves {
        if transb {
            let base = format!("wb + ({}u + tck) * (gd.n / 2u)", h * 8);
            let wo = "(n0 + 8u * tbog + 4u * tbos) / 2u";
            s.push_str(&format!(
                "  B4[({}u + tck) * BQ + tbop] = vec4<f32>(halve(W[{base} + {wo}], false), halve(W[{base} + {wo}], true), halve(W[{base} + {wo} + 1u], false), halve(W[{base} + {wo} + 1u], true));\n",
                h * 8
            ));
        } else {
            let mut comps = String::new();
            for e in 0..4 {
                comps.push_str(&format!(
                    " halve(W[wb + (n0 + 8u * bog + 4u * bos + {e}u) * kk + ({}u + bkl) / 2u], bodd),",
                    h * 8
                ));
            }
            s.push_str(&format!(
                "  B4[({}u + bkl) * BQ + bop] = vec4<f32>({comps});\n",
                h * 8
            ));
        }
    }
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= gd.k) { break; }\n  let kn = k0 + BK;\n");
    for e in 0..halves * 4 {
        s.push_str(&format!("  var pfa{e}: f32 = 0.0;\n"));
    }
    for e in 0..halves * 4 {
        s.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
    }
    s.push_str("  if (kn < gd.k) {\n");
    for h in 0..halves {
        for e in 0..4 {
            s.push_str(&format!(
                " pfa{} = A[abase + (m0 + 4u * arg + {e}u) * alda + (kn + {}u + akl)];\n",
                h * 4 + e,
                h * 8
            ));
        }
    }
    for h in 0..halves {
        if transb {
            let base = format!("wb + (kn + {}u + tck) * (gd.n / 2u)", h * 8);
            let wo = "(n0 + 8u * tbog + 4u * tbos) / 2u";
            s.push_str(&format!(
                "   pfb{} = W[{base} + {wo}];\n   pfb{} = W[{base} + {wo} + 1u];\n",
                h * 2,
                h * 2 + 1
            ));
        } else {
            for e in 0..4 {
                s.push_str(&format!(
                    "   pfb{} = W[wb + (n0 + 8u * bog + 4u * bos + {e}u) * kk + (kn + {}u + bkl) / 2u];\n",
                    h * 4 + e,
                    h * 8
                ));
            }
        }
    }
    s.push_str("  }\n");
    // compute: A4[(k)*AQ + row_group], B4[(k)*BQ + col_group]
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    for u in 0..4 {
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
    s.push_str("   q0 = q0 + 4u;\n  }\n");
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < gd.k) {\n");
    for h in 0..halves {
        let mut comps = String::new();
        for e in 0..4 {
            comps.push_str(&format!("   pfa{},", h * 4 + e));
        }
        s.push_str(&format!(
            "   A4[({}u + akl) * AQ + arg] = vec4<f32>({comps});\n",
            h * 8
        ));
    }
    for h in 0..halves {
        if transb {
            s.push_str(&format!(
                "   B4[({}u + tck) * BQ + tbop] = vec4<f32>(halve(pfb{}, false), halve(pfb{}, true), halve(pfb{}, false), halve(pfb{}, true));\n",
                h * 8,
                h * 2,
                h * 2,
                h * 2 + 1,
                h * 2 + 1
            ));
        } else {
            let mut comps = String::new();
            for e in 0..4 {
                comps.push_str(&format!("   halve(pfb{}, bodd),", h * 4 + e));
            }
            s.push_str(&format!(
                "   B4[({}u + bkl) * BQ + bop] = vec4<f32>({comps});\n",
                h * 8
            ));
        }
    }
    s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");
    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!(
                "C[cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {j}u] = c{i}{j};\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// 64x128 tile, 128 threads (8x8 per thread): half the registers/block and a
/// quarter of the shared memory of the 128x128 kernel, so ~5 blocks per SM
/// instead of 2.  Same K order and FMAs => bit-identical results.
pub fn gemm_mix_bm64(transb: bool) -> String {
    gemm_mix_bm64_param(transb, false)
}

/// 64x128 tile whose `W` operand is 4-bit codes plus a per-row level table
/// instead of f16 halves.
///
/// Weight layout in one storage buffer (single binding, so no call site changes):
///
/// ```text
/// [ levels: n_out rows x 4 f16 ][ codes: n_out rows x ceil(k/8) u32 ]
/// ```
///
/// Nibble `k & 7` of word `k >> 3` in a row is that weight's code, and the row's
/// first `k_lev` fp16 values are its levels: code `c` means `+level[c-1]` for
/// `1..=k_lev`, `-level[c-k_lev-1]` for `k_lev+1..=2*k_lev`, and `0` for `c == 0`.
/// That is exactly the mapping the reference loader applies, so the values the
/// kernel multiplies are the ones the float32 checkpoint would have held.
///
/// Every workgroup expands the 128 level entries it needs into an 8-slot table
/// in workgroup memory once, then each weight costs one code nibble plus one
/// table read in the load stage -- the same place the f16 kernel widens halves.
pub fn gemm_mix_bm64_lb() -> String {
    gemm_mix_bm64_param(false, true)
}

fn gemm_mix_bm64_param(transb: bool, lowbit: bool) -> String {
    let (tm, tn, bk, bm, bn) = (8usize, 8, 16, 64, 128);
    let (aq, bq) = (bm / 4 + 1, bn / 4 + 1);
    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\n\
         const TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\nconst TRANSB: u32 = {}u;\n",
        u32::from(transb)
    ));
    s.push_str(&format!(
        "var<workgroup> A4: array<vec4<f32>, {}>;\nvar<workgroup> B4: array<vec4<f32>, {}>;\n",
        bk * aq,
        bk * bq
    ));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n let p = unpack2x16float(w);\n return select(p.x, p.y, odd);\n}\n\
         fn swz(b: u32) -> u32 {\n return 8u * (((b >> 4u) << 3u) | (b & 7u)) + 4u * ((b >> 3u) & 1u);\n}\n",
    );
    s.push_str(
        "@compute @workgroup_size(16, 8)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         let c0 = ty * 16u + tx;\n\
         let arg = c0 / 16u;\n let akl = c0 % 16u;\n\
         let bop = c0 / 16u;\n let bkl = c0 % 16u;\n\
         let bodd = ((bkl & 1u) == 1u);\n\
         let tbop = c0 % 32u;\n let tbk = c0 / 32u;\n",
    );
    if lowbit {
        // Code base, this thread's nibble shift, and the tile's 128-row level
        // table (code -> value) expanded once into workgroup memory.
        s.push_str(
            "let wbc = wb + gd.lev_words + gd.n * 2u;\n let sh = 4u * (bkl & 7u);\n\
             let col = tx + 16u * ty;\n\
             let lw = gd.lev_words + (n0 + col) * 2u;\n\
             let l0 = halve(W[lw], false);\n let l1 = halve(W[lw], true);\n let l2 = halve(W[lw + 1u], false);\n\
             for (var c = 0u; c < 8u; c = c + 1u) {\n\
              var v = 0.0;\n\
              if (c > 0u && c <= 2u * gd.k_lev) {\n\
               let mag = select(c - gd.k_lev, c, c <= gd.k_lev) - 1u;\n\
               var lv = l2;\n\
               if (mag == 0u) { lv = l0; } else if (mag == 1u) { lv = l1; }\n\
               v = select(-lv, lv, c <= gd.k_lev);\n\
              }\n\
              TBL[col * 8u + c] = v;\n\
             }\n\
             workgroupBarrier();\n",
        );
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }
    // initial A: 2 elements (arg + 8*e), 4 rows each
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!(
                " A[abase + (m0 + 4u * ({arg}) + {r}u) * alda + akl],"
            ));
        }
        s.push_str(&format!("  A4[akl * AQ + ({arg})] = vec4<f32>({comps});\n"));
    }
    // initial B: 4 elements (bop + 8*e)
    for e in 0..4 {
        if transb {
            let k = format!("tbk + {}u", e * 4);
            let base = format!("wb + ({k}) * (gd.n / 2u)");
            let nb =
                "(n0 + 8u * (((tbop >> 4u) << 3u) | (tbop & 7u)) + 4u * ((tbop >> 3u) & 1u)) / 2u";
            s.push_str(&format!(
                "  B4[({k}) * BQ + tbop] = vec4<f32>(halve(W[{base} + {nb}], false), halve(W[{base} + {nb}], true), halve(W[{base} + {nb} + 1u], false), halve(W[{base} + {nb} + 1u], true));\n"
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            let mut comps = String::new();
            for r in 0..4 {
                if lowbit {
                    // one code nibble + one table read, instead of widening a half
                    comps.push_str(&format!(
                        " TBL[(swz({bop}) + {r}u) * 8u + ((W[wbc + (n0 + swz({bop}) + {r}u) * gd.codes_ld + bkl / 8u] >> sh) & 15u)],"
                    ));
                } else {
                    comps.push_str(&format!(
                        " halve(W[wb + (n0 + swz({bop}) + {r}u) * kk + (0u + bkl) / 2u], bodd),"
                    ));
                }
            }
            s.push_str(&format!("  B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"));
        }
    }
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= gd.k) { break; }\n  let kn = k0 + BK;\n");
    for e in 0..8 {
        s.push_str(&format!("  var pfa{e}: f32 = 0.0;\n"));
    }
    for e in 0..16 {
        s.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
    }
    s.push_str("  if (kn < gd.k) {\n");
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        for r in 0..4 {
            s.push_str(&format!(
                "   pfa{} = A[abase + (m0 + 4u * ({arg}) + {r}u) * alda + (kn + akl)];\n",
                e * 4 + r
            ));
        }
    }
    for e in 0..4 {
        if transb {
            let k = format!("tbk + {}u", e * 4);
            let base = format!("wb + (kn + {k}) * (gd.n / 2u)");
            let nb =
                "(n0 + 8u * (((tbop >> 4u) << 3u) | (tbop & 7u)) + 4u * ((tbop >> 3u) & 1u)) / 2u";
            s.push_str(&format!(
                "   pfb{} = W[{base} + {nb}];\n   pfb{} = W[{base} + {nb} + 1u];\n",
                e * 4,
                e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            for r in 0..4 {
                if lowbit {
                    s.push_str(&format!(
                        "   pfb{} = W[wbc + (n0 + swz({bop}) + {r}u) * gd.codes_ld + (kn + bkl) / 8u];\n",
                        e * 4 + r
                    ));
                } else {
                    s.push_str(&format!(
                        "   pfb{} = W[wb + (n0 + swz({bop}) + {r}u) * kk + (kn + bkl) / 2u];\n",
                        e * 4 + r
                    ));
                }
            }
        }
    }
    s.push_str("  }\n");
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    for u in 0..4 {
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
    s.push_str("   q0 = q0 + 4u;\n  }\n");
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < gd.k) {\n");
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!("   pfa{},", e * 4 + r));
        }
        s.push_str(&format!(
            "   A4[akl * AQ + ({arg})] = vec4<f32>({comps});\n"
        ));
    }
    for e in 0..4 {
        if transb {
            let k = format!("tbk + {}u", e * 4);
            s.push_str(&format!(
                "   B4[({k}) * BQ + tbop] = vec4<f32>(halve(pfb{}, false), halve(pfb{}, true), halve(pfb{}, false), halve(pfb{}, true));\n",
                e * 4, e * 4, e * 4 + 1, e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            let mut comps = String::new();
            for r in 0..4 {
                if lowbit {
                    comps.push_str(&format!(
                        "   TBL[(swz({bop}) + {r}u) * 8u + ((pfb{} >> sh) & 15u)],",
                        e * 4 + r
                    ));
                } else {
                    comps.push_str(&format!("   halve(pfb{}, bodd),", e * 4 + r));
                }
            }
            s.push_str(&format!(
                "   B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"
            ));
        }
    }
    s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");
    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!(
                "C[cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {j}u] = c{i}{j};\n"
            ));
        }
    }
    s.push_str("}\n");
    if lowbit {
        s = s.replacen(
            "row0: u32, lda: u32 };",
            "row0: u32, lda: u32, _pad: u32,\n         lev_words: u32, k_lev: u32, codes_ld: u32, lowbit: u32 };",
            1,
        );
        s = s.replacen(
            "var<workgroup> B4:",
            "var<workgroup> TBL: array<f32, 1024>;\nvar<workgroup> B4:",
            1,
        );
    }
    s
}

/// 64x64 tile, 128 threads, 8 rows x 4 cols per thread (32 accumulators).
///
/// Same shared-memory layout discipline and the same per-output single
/// accumulator over ascending k as `gemm_mix_bm64`, so the f32 results are
/// bit-identical; only the tiling (and therefore the register/occupancy
/// trade-off) differs: half the accumulators of the 8x8 kernel and 1/4 the
/// shared memory of the 64x128 kernel (two 272-vec4 tiles), at the cost of
/// re-reading the A tile for every 64 columns instead of every 128.
pub fn gemm_mix_8x4(transb: bool) -> String {
    let (tm, tn, bk, bm, bn) = (8usize, 4, 16, 64, 64);
    let (aq, bq) = (bm / 4 + 1, bn / 4 + 1);
    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\n\
         const TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\nconst TRANSB: u32 = {}u;\n",
        u32::from(transb)
    ));
    s.push_str(&format!(
        "var<workgroup> A4: array<vec4<f32>, {}>;\nvar<workgroup> B4: array<vec4<f32>, {}>;\n",
        bk * aq,
        bk * bq
    ));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n let p = unpack2x16float(w);\n return select(p.x, p.y, odd);\n}\n",
    );
    // Column-group indices map straight onto n (group g owns columns 4g..4g+3),
    // so both the staged stores and the C stores stay contiguous per warp.
    s.push_str(
        "@compute @workgroup_size(16, 8)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         let c0 = ty * 16u + tx;\n\
         let arg = c0 / 16u;\n let akl = c0 % 16u;\n\
         let bop = c0 / 16u;\n let bkl = c0 % 16u;\n\
         let bodd = ((bkl & 1u) == 1u);\n\
         let tbop = c0 % 16u;\n let tbk = c0 / 16u;\n",
    );
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }
    // A: 2 elements per thread (2 of the 16 row groups, at its own k).
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!(
                " A[abase + (m0 + 4u * ({arg}) + {r}u) * alda + akl],"
            ));
        }
        s.push_str(&format!("  A4[akl * AQ + ({arg})] = vec4<f32>({comps});\n"));
    }
    // B: 2 elements per thread (2 of the 16 column groups, at its own k).
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            let base = format!("wb + ({k}) * (gd.n / 2u)");
            let nb = "(n0 + 4u * tbop) / 2u";
            s.push_str(&format!(
                "  B4[({k}) * BQ + tbop] = vec4<f32>(halve(W[{base} + {nb}], false), halve(W[{base} + {nb}], true), halve(W[{base} + {nb} + 1u], false), halve(W[{base} + {nb} + 1u], true));\n"
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            let mut comps = String::new();
            for r in 0..4 {
                comps.push_str(&format!(
                    " halve(W[wb + (n0 + 4u * ({bop}) + {r}u) * kk + (0u + bkl) / 2u], bodd),"
                ));
            }
            s.push_str(&format!("  B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"));
        }
    }
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= gd.k) { break; }\n  let kn = k0 + BK;\n");
    for e in 0..8 {
        s.push_str(&format!("  var pfa{e}: f32 = 0.0;\n"));
    }
    for e in 0..8 {
        s.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
    }
    s.push_str("  if (kn < gd.k) {\n");
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        for r in 0..4 {
            s.push_str(&format!(
                "   pfa{} = A[abase + (m0 + 4u * ({arg}) + {r}u) * alda + (kn + akl)];\n",
                e * 4 + r
            ));
        }
    }
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            let base = format!("wb + (kn + {k}) * (gd.n / 2u)");
            let nb = "(n0 + 4u * tbop) / 2u";
            s.push_str(&format!(
                "   pfb{} = W[{base} + {nb}];\n   pfb{} = W[{base} + {nb} + 1u];\n",
                e * 4,
                e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            for r in 0..4 {
                s.push_str(&format!(
                    "   pfb{} = W[wb + (n0 + 4u * ({bop}) + {r}u) * kk + (kn + bkl) / 2u];\n",
                    e * 4 + r
                ));
            }
        }
    }
    s.push_str("  }\n");
    // Ascending k, one accumulator per output: identical FMA order to bm64.
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    for u in 0..4 {
        let idx = format!("(q0 + {u}u)");
        s.push_str(&format!("   let av_{u} = A4[{idx} * AQ + ty * 2u];\n"));
        s.push_str(&format!("   let aw_{u} = A4[{idx} * AQ + ty * 2u + 1u];\n"));
        s.push_str(&format!("   let bv_{u} = B4[{idx} * BQ + tx];\n"));
        for i in 0..tm {
            let a = if i < 4 {
                format!("av_{u}[{i}]")
            } else {
                format!("aw_{u}[{}]", i - 4)
            };
            for j in 0..tn {
                s.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * bv_{u}[{j}];\n"));
            }
        }
    }
    s.push_str("   q0 = q0 + 4u;\n  }\n");
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < gd.k) {\n");
    for e in 0..2 {
        let arg = format!("arg + {}u", e * 8);
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!("   pfa{},", e * 4 + r));
        }
        s.push_str(&format!(
            "   A4[akl * AQ + ({arg})] = vec4<f32>({comps});\n"
        ));
    }
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            s.push_str(&format!(
                "   B4[({k}) * BQ + tbop] = vec4<f32>(halve(pfb{}, false), halve(pfb{}, true), halve(pfb{}, false), halve(pfb{}, true));\n",
                e * 4, e * 4, e * 4 + 1, e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 8);
            let mut comps = String::new();
            for r in 0..4 {
                comps.push_str(&format!("   halve(pfb{}, bodd),", e * 4 + r));
            }
            s.push_str(&format!(
                "   B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"
            ));
        }
    }
    s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");
    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!(
                "C[cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {j}u] = c{i}{j};\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// 64x128 tile, 256 threads, 4x8 per thread (32 accumulators).  Half the
/// registers of the 8x8 kernel -> ~40 resident warps/SM instead of 20.
pub fn gemm_mix_4x8(transb: bool) -> String {
    let (tm, tn, bk, bm, bn) = (4usize, 8, 16, 64, 128);
    let (aq, bq) = (bm / 4 + 1, bn / 4 + 1);
    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<f32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst TRANSB: u32 = {}u;\n",
        u32::from(transb)
    ));
    s.push_str(&format!(
        "var<workgroup> A4: array<vec4<f32>, {}>;\nvar<workgroup> B4: array<vec4<f32>, {}>;\n",
        bk * aq,
        bk * bq
    ));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n let p = unpack2x16float(w);\n return select(p.x, p.y, odd);\n}\n\
         fn swz(b: u32) -> u32 {\n return 8u * (((b >> 4u) << 3u) | (b & 7u)) + 4u * ((b >> 3u) & 1u);\n}\n",
    );
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         let c0 = ty * 16u + tx;\n\
         let arg = c0 / 16u;\n let akl = c0 % 16u;\n\
         let bop = c0 / 16u;\n let bkl = c0 % 16u;\n\
         let bodd = ((bkl & 1u) == 1u);\n\
         let tbop = c0 % 32u;\n let tbk = c0 / 32u;\n",
    );
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }
    // A: 1 element per thread (16 row groups x 16 k = 256)
    {
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!(" A[abase + (m0 + 4u * arg + {r}u) * alda + akl],"));
        }
        s.push_str(&format!("  A4[akl * AQ + arg] = vec4<f32>({comps});\n"));
    }
    // B: 2 elements per thread
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            let base = format!("wb + ({k}) * (gd.n / 2u)");
            let nb =
                "(n0 + 8u * (((tbop >> 4u) << 3u) | (tbop & 7u)) + 4u * ((tbop >> 3u) & 1u)) / 2u";
            s.push_str(&format!(
                "  B4[({k}) * BQ + tbop] = vec4<f32>(halve(W[{base} + {nb}], false), halve(W[{base} + {nb}], true), halve(W[{base} + {nb} + 1u], false), halve(W[{base} + {nb} + 1u], true));\n"
            ));
        } else {
            let bop = format!("bop + {}u", e * 16);
            let mut comps = String::new();
            for r in 0..4 {
                comps.push_str(&format!(
                    " halve(W[wb + (n0 + swz({bop}) + {r}u) * kk + (0u + bkl) / 2u], bodd),"
                ));
            }
            s.push_str(&format!("  B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"));
        }
    }
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= gd.k) { break; }\n  let kn = k0 + BK;\n");
    for e in 0..4 {
        s.push_str(&format!("  var pfa{e}: f32 = 0.0;\n"));
    }
    for e in 0..8 {
        s.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
    }
    s.push_str("  if (kn < gd.k) {\n");
    for r in 0..4 {
        s.push_str(&format!(
            "   pfa{r} = A[abase + (m0 + 4u * arg + {r}u) * alda + (kn + akl)];\n"
        ));
    }
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            let base = format!("wb + (kn + {k}) * (gd.n / 2u)");
            let nb =
                "(n0 + 8u * (((tbop >> 4u) << 3u) | (tbop & 7u)) + 4u * ((tbop >> 3u) & 1u)) / 2u";
            s.push_str(&format!(
                "   pfb{} = W[{base} + {nb}];\n   pfb{} = W[{base} + {nb} + 1u];\n",
                e * 4,
                e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 16);
            for r in 0..4 {
                s.push_str(&format!(
                    "   pfb{} = W[wb + (n0 + swz({bop}) + {r}u) * kk + (kn + bkl) / 2u];\n",
                    e * 4 + r
                ));
            }
        }
    }
    s.push_str("  }\n");
    s.push_str("  var q0: u32 = 0u;\n  loop {\n   if (q0 >= BK) { break; }\n");
    for u in 0..4 {
        let idx = format!("(q0 + {u}u)");
        s.push_str(&format!("   let av_{u} = A4[{idx} * AQ + ty];\n"));
        s.push_str(&format!(
            "   let bv_{u} = B4[{idx} * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n"
        ));
        s.push_str(&format!(
            "   let bw_{u} = B4[{idx} * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
        ));
        for i in 0..tm {
            let a = format!("av_{u}[{i}]");
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
    s.push_str("   q0 = q0 + 4u;\n  }\n");
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < gd.k) {\n");
    {
        let mut comps = String::new();
        for r in 0..4 {
            comps.push_str(&format!("   pfa{r},"));
        }
        s.push_str(&format!("   A4[akl * AQ + arg] = vec4<f32>({comps});\n"));
    }
    for e in 0..2 {
        if transb {
            let k = format!("tbk + {}u", e * 8);
            s.push_str(&format!(
                "   B4[({k}) * BQ + tbop] = vec4<f32>(halve(pfb{}, false), halve(pfb{}, true), halve(pfb{}, false), halve(pfb{}, true));\n",
                e * 4, e * 4, e * 4 + 1, e * 4 + 1
            ));
        } else {
            let bop = format!("bop + {}u", e * 16);
            let mut comps = String::new();
            for r in 0..4 {
                comps.push_str(&format!("   halve(pfb{}, bodd),", e * 4 + r));
            }
            s.push_str(&format!(
                "   B4[bkl * BQ + ({bop})] = vec4<f32>({comps});\n"
            ));
        }
    }
    s.push_str("  }\n  workgroupBarrier();\n  k0 = kn;\n }\n");
    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!(
                "C[cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {j}u] = c{i}{j};\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}
