use anyhow::Result;
use bark3_pro_wgpu::encoder::{Encoder, Weight};
use bark3_pro_wgpu::raw::RawTensor;

pub fn gemm_mix_z8_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let g = &enc.gpu;
    let tp = pad128(191);
    let (m, n, k, z) = (tp as usize, 128usize, tp as usize, 8usize);
    // A (z, m, k) f32; W (z, k, n) f16.
    let af32: Vec<f32> = (0..z * m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let wf32: Vec<f32> = (0..z * k * n)
        .map(|i| (((i * 31 + 7) % 113) as f32 - 56.0) * 0.01)
        .collect();
    let w_words = RawTensor::f32_to_f16_words(&wf32);
    let a = g.storage("gz_a", af32.len() as u64 * 4);
    let w = g.storage("gz_w", w_words.len() as u64 * 4);
    let c = g.storage("gz_c", z as u64 * m as u64 * n as u64 * 4);
    g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&af32));
    g.queue.write_buffer(&w, 0, bytemuck::cast_slice(&w_words));
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm_f32(
            &mut cp,
            true,
            &a,
            &Weight::f16(w.clone()),
            &c,
            m as u32,
            n as u32,
            k as u32,
            n as u32,
            (m * k) as u32,
            (k * n / 2) as u32,
            (m * n) as u32,
            k as u32,
            z as u32,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&c, z as u64 * m as u64 * n as u64 * 4)?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let wh = unpack_weights(&w_words);
    let mut max_diff = 0f32;
    let mut mean_diff = 0f64;
    let mut worst = (0usize, 0usize);
    for batch in 0..z {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += af32[batch * m * k + row * k + i] * wh[batch * k * n + i * n + col];
                }
                let diff = (got[batch * m * n + row * n + col] - acc).abs();
                if diff > max_diff {
                    max_diff = diff;
                    worst = (batch, row);
                }
                mean_diff += diff as f64;
            }
        }
    }
    println!(
        "gemm_mix transb z8: max|D|={max_diff:.5} mean|D|={:.6} worst_batch_row={worst:?}",
        mean_diff / (z * m * n) as f64
    );
    Ok(())
}

pub fn gemm_mix_tb_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let g = &enc.gpu;
    let tp = pad128(191);
    let (m, n, k) = (tp as usize, 128usize, tp as usize);
    // A (m, k) f32; W stored [k, n] f16 (row-major k rows of n).
    let af32: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let wf32: Vec<f32> = (0..k * n)
        .map(|i| (((i * 31 + 7) % 113) as f32 - 56.0) * 0.01)
        .collect();
    let w_words = RawTensor::f32_to_f16_words(&wf32);
    let a = g.storage("gt_a", af32.len() as u64 * 4);
    let w = g.storage("gt_w", w_words.len() as u64 * 4);
    let c = g.storage("gt_c", m as u64 * n as u64 * 4);
    g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&af32));
    g.queue.write_buffer(&w, 0, bytemuck::cast_slice(&w_words));
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm_f32(
            &mut cp,
            true,
            &a,
            &Weight::f16(w.clone()),
            &c,
            m as u32,
            n as u32,
            k as u32,
            n as u32,
            0,
            0,
            0,
            k as u32,
            1,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&c, (m * n) as u64 * 4)?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let wh = unpack_weights(&w_words);
    let mut max_diff = 0f32;
    let mut mean_diff = 0f64;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += af32[row * k + i] * wh[i * n + col];
            }
            let diff = (got[row * n + col] - acc).abs();
            max_diff = max_diff.max(diff);
            mean_diff += diff as f64;
        }
    }
    println!(
        "gemm_mix transb m={m} n={n} k={k}: max|D|={max_diff:.5} mean|D|={:.6}",
        mean_diff / (m * n) as f64
    );
    if std::env::var("GMIX_8X4").is_ok() && m % 64 == 0 && n % 64 == 0 {
        let c8 = g.storage("gt_c8", (m * n) as u64 * 4);
        let mut e8 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e8.begin_compute_pass(&Default::default());
            ctx.gemm_f32_8x4(
                &mut cp, true, &a, &w, &c8, m as u32, n as u32, k as u32, n as u32, 0, 0, 0,
                k as u32, 1,
            );
        }
        g.queue.submit([e8.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        let bytes8 = g.readback(&c8, (m * n) as u64 * 4)?;
        let got8: Vec<f32> = bytemuck::cast_slice(&bytes8).to_vec();
        let mut bit_diff = 0usize;
        let mut max_d_bm64 = 0f32;
        for i in 0..m * n {
            if got8[i].to_bits() != got[i].to_bits() {
                bit_diff += 1;
                max_d_bm64 = max_d_bm64.max((got8[i] - got[i]).abs());
            }
        }
        println!(
            "gemm_mix transb 8x4 m={m} n={n} k={k}: vs bm64 bit_mismatch={bit_diff}/{} max|D|={max_d_bm64:.6}",
            m * n
        );
    }
    Ok(())
}

pub fn gemm_mix_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let g = &enc.gpu;
    let tp = pad128(191);
    let (m, n, k) = (tp as usize, 1152usize, 128usize);
    let (m, n, k, z, bsa, bsb, bsc, lda) = if let Ok(spec) = std::env::var("GMIX_SHAPE") {
        let values: Vec<usize> = spec.split(',').map(|x| x.trim().parse().unwrap()).collect();
        (
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        )
    } else {
        (m, n, k, 1, 0, 0, 0, k)
    };
    let af32: Vec<f32> = (0..z * m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let wf32: Vec<f32> = (0..z * n * k)
        .map(|i| (((i * 31 + 7) % 113) as f32 - 56.0) * 0.01)
        .collect();
    let w_words = RawTensor::f32_to_f16_words(&wf32);
    let a = g.storage("gm_a", af32.len() as u64 * 4);
    let w = g.storage("gm_w", w_words.len() as u64 * 4);
    let c = g.storage("gm_c", z as u64 * m as u64 * n as u64 * 4);
    g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&af32));
    g.queue.write_buffer(&w, 0, bytemuck::cast_slice(&w_words));
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm_f32(
            &mut cp,
            false,
            &a,
            &Weight::f16(w.clone()),
            &c,
            m as u32,
            n as u32,
            k as u32,
            n as u32,
            bsa as u32,
            bsb as u32,
            bsc as u32,
            lda as u32,
            z as u32,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&c, (z * m * n) as u64 * 4)?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let wh = unpack_weights(&w_words);
    let mut max_diff = 0f32;
    let mut mean_diff = 0f64;
    for batch in 0..z {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += af32[batch * bsa + row * lda + i] * wh[batch * (bsb * 2) + col * k + i];
                }
                let diff = (got[batch * bsc + row * n + col] - acc).abs();
                max_diff = max_diff.max(diff);
                mean_diff += diff as f64;
            }
        }
    }
    println!(
        "gemm_mix m={m} n={n} k={k} z={z}: max|D|={max_diff:.5} mean|D|={:.6}",
        mean_diff / (m * n) as f64
    );
    if std::env::var("GMIX_8X4").is_ok() {
        if m % 64 != 0 || n % 64 != 0 {
            println!("gemm_mix 8x4: skipped (m={m} n={n} not multiples of 64)");
            return Ok(());
        }
        let c8 = g.storage("gm_c8", z as u64 * m as u64 * n as u64 * 4);
        let mut e8 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e8.begin_compute_pass(&Default::default());
            ctx.gemm_f32_8x4(
                &mut cp, false, &a, &w, &c8, m as u32, n as u32, k as u32, n as u32, bsa as u32,
                bsb as u32, bsc as u32, lda as u32, z as u32,
            );
        }
        g.queue.submit([e8.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        let bytes8 = g.readback(&c8, (z * m * n) as u64 * 4)?;
        let got8: Vec<f32> = bytemuck::cast_slice(&bytes8).to_vec();
        // vs the production bm64 kernel: expected to be bit-identical (same k order).
        let mut bit_diff = 0usize;
        let mut max_d_bm64 = 0f32;
        // vs the f32 CPU reference.
        let mut max_d_cpu = 0f32;
        let mut mean_d_cpu = 0f64;
        for batch in 0..z {
            for row in 0..m {
                for col in 0..n {
                    let i = batch * m * n + row * n + col;
                    if got8[i].to_bits() != got[i].to_bits() {
                        bit_diff += 1;
                        max_d_bm64 = max_d_bm64.max((got8[i] - got[i]).abs());
                    }
                    let mut acc = 0f32;
                    for t in 0..k {
                        acc +=
                            af32[batch * bsa + row * lda + t] * wh[batch * (bsb * 2) + col * k + t];
                    }
                    let d = (got8[i] - acc).abs();
                    max_d_cpu = max_d_cpu.max(d);
                    mean_d_cpu += d as f64;
                }
            }
        }
        println!(
            "gemm_mix 8x4 m={m} n={n} k={k} z={z}: vs bm64 bit_mismatch={bit_diff}/{} max|D|={max_d_bm64:.6} | vs cpu max|D|={max_d_cpu:.5} mean|D|={:.6}",
            z * m * n,
            mean_d_cpu / (z * m * n) as f64
        );
    }
    Ok(())
}

pub fn gemm_bench(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::Ctx;
    let g = &enc.gpu;
    let reps: usize = std::env::var("GB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let shapes: Vec<(u32, u32, u32, u32)> = vec![
        (2304, 4096, 1024, 1),
        (2304, 1024, 4096, 1),
        (2304, 3072, 1024, 1),
        (4480, 1024, 1024, 1),
        (2304, 2304, 128, 8),
        (2304, 4480, 128, 8),
    ];
    for (m, n, k, z) in shapes {
        let a = g.storage("gb_a", m as u64 * k as u64 * z as u64 * 4);
        let w = g.storage("gb_w", n as u64 * k as u64 * z as u64 * 2);
        let c = g.storage("gb_c", m as u64 * n as u64 * z as u64 * 4);
        let av = vec![0.01f32; (m * k * z) as usize];
        g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&av));
        let wv = vec![0u32; ((n * k * z) as usize).div_ceil(2)];
        g.queue.write_buffer(&w, 0, bytemuck::cast_slice(&wv));
        let mut ctx = Ctx::new(enc, 0);
        let mut e2 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e2.begin_compute_pass(&Default::default());
            let bm64 = std::env::var("GB_BM64").is_ok();
            let g4x8 = std::env::var("GB_4X8").is_ok();
            let g8x4 = std::env::var("GB_8X4").is_ok();
            let g128 = std::env::var("GB_128").is_ok();
            for _ in 0..reps {
                if g8x4 {
                    ctx.gemm_f32_8x4(
                        &mut cp,
                        false,
                        &a,
                        &w,
                        &c,
                        m,
                        n,
                        k,
                        n,
                        m * k,
                        k * n / 2,
                        m * n,
                        k,
                        z,
                    );
                } else if g4x8 {
                    ctx.gemm_f32_4x8(
                        &mut cp,
                        false,
                        &a,
                        &w,
                        &c,
                        m,
                        n,
                        k,
                        n,
                        m * k,
                        k * n / 2,
                        m * n,
                        k,
                        z,
                    );
                } else if g128 {
                    ctx.gemm_f32_128(
                        &mut cp,
                        false,
                        &a,
                        &w,
                        &c,
                        m,
                        n,
                        k,
                        n,
                        m * k,
                        k * n / 2,
                        m * n,
                        k,
                        z,
                    );
                } else if bm64 {
                    ctx.gemm_f32_bm64(
                        &mut cp,
                        false,
                        &a,
                        &w,
                        &c,
                        m,
                        n,
                        k,
                        n,
                        m * k,
                        k * n / 2,
                        m * n,
                        k,
                        z,
                    );
                } else {
                    ctx.gemm_f32(
                        &mut cp,
                        false,
                        &a,
                        &Weight::f16(w.clone()),
                        &c,
                        m,
                        n,
                        k,
                        n,
                        m * k,
                        k * n / 2,
                        m * n,
                        k,
                        z,
                    );
                }
            }
        }
        let started = std::time::Instant::now();
        g.queue.submit([e2.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = started.elapsed().as_secs_f64();
        let flops = 2.0 * m as f64 * n as f64 * k as f64 * z as f64 * reps as f64;
        println!(
            "gemm {}x{}x{} z={} : {:.1} ms/call  {:.2} TFLOPS",
            m,
            n,
            k,
            z,
            elapsed * 1000.0 / reps as f64,
            flops / elapsed / 1e12
        );
    }
    Ok(())
}

pub fn gemm_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let g = &enc.gpu;
    let tp = pad128(191);
    let burn: usize = std::env::var("GEMM_BURN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for (n, k) in [
        (1024usize, 1024usize),
        (1024, 4096),
        (4096, 4096),
        (4096, 1024),
        (3072, 1024),
        (2048, 1024),
    ] {
        let real_w = ((n, k) == (4096, 1024) || (n, k) == (2048, 1024) || (n, k) == (1024, 4096))
            && std::env::var("GEMM_REAL_W").is_ok();
        let wf32: Vec<f32> = if real_w {
            read_real_weights(enc, n, k)?
        } else {
            (0..n * k)
                .map(|i| (((i * 31 + 7) % 113) as f32 - 56.0) * 0.01)
                .collect()
        };
        let w_words = RawTensor::f32_to_f16_words(&wf32);
        let af32: Vec<f32> = (0..tp * k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let a_words = RawTensor::f32_to_f16_words(&af32);
        let a = g.storage("gt_a", a_words.len() as u64 * 4);
        let w = g.storage("gt_w", w_words.len() as u64 * 4);
        let c = g.storage("gt_c", tp as u64 * n as u64 * 2);
        g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&a_words));
        g.queue.write_buffer(&w, 0, bytemuck::cast_slice(&w_words));
        let mut ctx = Ctx::new(enc, 0);
        for _ in 0..burn {
            let _ = ctx.next_slot(&[0u8; 4]);
        }
        let mut e2 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e2.begin_compute_pass(&Default::default());
            ctx.gemm(
                &mut cp, false, &a, &w, &c, tp as u32, n as u32, k as u32, n as u32, 0, 0, 0,
                k as u32, 1,
            );
        }
        g.queue.submit([e2.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        let bytes = g.readback(&c, tp as u64 * n as u64 * 2)?;
        let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let ah = unpack_weights(&a_words);
        let wh = unpack_weights(&w_words);
        let mut max_diff = 0f32;
        let mut mean_diff = 0f64;
        let mut count = 0;
        for row in 0..tp {
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += ah[row * k + i] * wh[col * k + i];
                }
                let diff = (got[row * n + col] - acc).abs();
                max_diff = max_diff.max(diff);
                mean_diff += diff as f64;
                count += 1;
            }
        }
        println!(
            "n={n:>5} k={k:>5}: max|Δ|={max_diff:.4} mean|Δ|={:.5} ({} elems){}",
            mean_diff / count as f64,
            count,
            if real_w { " [REAL_W]" } else { "" }
        );
    }
    Ok(())
}

fn unpack_weights(words: &[u32]) -> Vec<f32> {
    bytemuck::cast_slice::<u32, [u16; 2]>(words)
        .iter()
        .flat_map(|pair| pair.iter())
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .collect()
}

/// Weights already resident on the device, read back as f32.
///
/// The bench wants realistic magnitudes, not a file path: the checkpoint is
/// already uploaded by the time a diagnostic runs, so this reuses layer 0.
fn read_real_weights(enc: &Encoder, n: usize, k: usize) -> Result<Vec<f32>> {
    let lw = &enc.w.layers[0];
    let w = match (n, k) {
        (4096, 1024) => &lw.ff1a,
        (2048, 1024) => &lw.pw1,
        (1024, 4096) => &lw.ff1b,
        _ => anyhow::bail!("no resident weight has shape {n}x{k}"),
    };
    anyhow::ensure!(
        w.lb.is_none(),
        "GEMM_REAL_W needs an f16 checkpoint; this one stores low-bit codes"
    );
    let bytes = enc.gpu.readback(&w.buf, (n * k * 2) as u64)?;
    let words: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
    Ok(unpack_weights(&words))
}

/// Low-bit (`A*W` with 4-bit codes + levels) against the f16 kernel on the same
/// values: the dequant must produce bit-identical f32 results.
pub fn lowbit_vs_f16_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, LowBit};

    let g = &enc.gpu;
    let (m, n, k) = (64usize, 128usize, 1024usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.002)
        .collect();

    // Two levels per row; codes 0..4 => 0, +l0, +l1, -l0, -l1.
    let levels: Vec<[f32; 2]> = (0..n)
        .map(|row| [0.02 * (row % 7 + 1) as f32, 0.05 * (row % 5 + 1) as f32])
        .collect();
    let codes: Vec<u8> = (0..n * k).map(|i| ((i * 13 + i / k) % 5) as u8).collect();
    let value = |row: usize, j: usize| -> f32 {
        let c = codes[row * k + j];
        let [l0, l1] = levels[row];
        match c {
            0 => 0.0,
            1 => l0,
            2 => l1,
            3 => -l0,
            _ => -l1,
        }
    };
    let wf32: Vec<f32> = (0..n * k).map(|i| value(i / k, i % k)).collect();

    // f16 layout: row-major halves, two per u32.
    let w_words = RawTensor::f32_to_f16_words(&wf32);
    let w16 = g.storage("lb_w16", w_words.len() as u64 * 4);
    g.queue
        .write_buffer(&w16, 0, bytemuck::cast_slice(&w_words));

    // Low-bit layout: [levels: n x 4 f16][codes: n x k/8 u32], nibble k&7.
    let mut lev_words = vec![0u32; n * 2];
    for row in 0..n {
        for (i, l) in levels[row].iter().enumerate() {
            let h = bark3_pro_wgpu::raw::f32_to_f16_bits(*l) as u32;
            lev_words[row * 2 + i / 2] |= h << (16 * (i % 2));
        }
    }
    let codes_ld = k / 8;
    let mut code_words = vec![0u32; n * codes_ld];
    for row in 0..n {
        for j in 0..k {
            code_words[row * codes_ld + j / 8] |= (codes[row * k + j] as u32) << (4 * (j % 8));
        }
    }
    let lb_len = (lev_words.len() + code_words.len()) as u64 * 4;
    let wlb = g.storage("lb_wlb", lb_len);
    g.queue
        .write_buffer(&wlb, 0, bytemuck::cast_slice(&lev_words));
    g.queue.write_buffer(
        &wlb,
        lev_words.len() as u64 * 4,
        bytemuck::cast_slice(&code_words),
    );

    let abuf = g.storage("lb_a", (m * k * 4) as u64);
    g.queue.write_buffer(&abuf, 0, bytemuck::cast_slice(&a));
    let c16 = g.storage("lb_c16", (m * n * 4) as u64);
    let clb = g.storage("lb_clb", (m * n * 4) as u64);

    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        for (w, c) in [
            (Weight::f16(w16.clone()), &c16),
            (
                Weight {
                    buf: wlb.clone(),
                    lb: Some(LowBit {
                        lev_words: 0,
                        k_lev: 2,
                        codes_ld: codes_ld as u32,
                    }),
                },
                &clb,
            ),
        ] {
            ctx.gemm_f32(
                &mut cp, false, &abuf, &w, c, m as u32, n as u32, k as u32, n as u32, 0, 0, 0,
                k as u32, 1,
            );
        }
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;

    let a16: Vec<f32> = bytemuck::cast_slice(&g.readback(&c16, (m * n * 4) as u64)?).to_vec();
    let alb: Vec<f32> = bytemuck::cast_slice(&g.readback(&clb, (m * n * 4) as u64)?).to_vec();
    let mut diff = 0f32;
    for i in 0..m * n {
        diff = diff.max((a16[i] - alb[i]).abs());
    }
    // Reference for the low-bit values (f32 accumulation, same k order).
    let mut ref_max = 0f32;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for j in 0..k {
                acc += a[row * k + j] * wf32[col * k + j];
            }
            ref_max = ref_max.max((acc - alb[row * n + col]).abs());
        }
    }
    println!(
        "[lbtest] f16 vs low-bit max|Δ| = {diff:e}; low-bit vs reference max|Δ| = {ref_max:e}"
    );
    println!("[lbtest] sample f16 {:?}", &a16[..3]);
    println!("[lbtest] sample low {:?}", &alb[..3]);
    anyhow::ensure!(diff == 0.0, "low-bit kernel disagrees with f16: {diff:e}");
    Ok(())
}
