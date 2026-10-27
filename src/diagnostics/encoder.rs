use anyhow::Result;
use bark3_pro_wgpu::encoder::Encoder;

use super::{diff_report, read_f32};

pub fn f2h_mix_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let g = &enc.gpu;
    // qkv buffer f32 (tp, 3072): upload torch_normed_sa, run gemm, then f2h_mix.
    let mut normed_pad = vec![0f32; tp as usize * 1024];
    normed_pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_normed_sa.bin")));
    let normed = g.storage("f2_n", normed_pad.len() as u64 * 4);
    g.queue
        .write_buffer(&normed, 0, bytemuck::cast_slice(&normed_pad));
    let qkv = g.storage("f2_q", tp as u64 * 3072 * 4);
    let kh = g.storage("f2_k", tp as u64 * 512 * 4);
    let vh = g.storage("f2_v", tp as u64 * 512 * 4);
    let lw = &enc.w.layers[0];
    let w512 = tp * 512;
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm_f32(
            &mut cp, false, &normed, &lw.qkv, &qkv, tp, 3072, 1024, 3072, 0, 0, 0, 1024, 1,
        );
        ctx.f2h_mix(&mut cp, &qkv, &kh, tp, w512, 3072, 1024);
        ctx.f2h_mix(&mut cp, &qkv, &vh, tp, w512, 3072, 2048);
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&kh, tp as u64 * 512 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let want = read_f32(&golden.join("a32_kh.bin"));
    diff_report("f2h_mix_kh", &got, &want);
    let bytes = g.readback(&vh, tp as u64 * 512 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let want = read_f32(&golden.join("a32_vh.bin"));
    diff_report("f2h_mix_vh", &got, &want);
    Ok(())
}

pub fn ff1a_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let g = &enc.gpu;
    let mut pad = vec![0f32; tp as usize * 1024];
    pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_normed0.bin")));
    let a = g.storage("fa_a", pad.len() as u64 * 4);
    g.queue.write_buffer(&a, 0, bytemuck::cast_slice(&pad));
    let c = g.storage("fa_c", tp as u64 * 4096 * 4);
    let weight_bytes = g.readback(&enc.w.layers[0].ff1a.buf, 16)?;
    let weights: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&weight_bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    println!("ff1a w[0..4] = {weights:?}");
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm_f32(
            &mut cp,
            false,
            &a,
            &enc.w.layers[0].ff1a,
            &c,
            tp,
            4096,
            1024,
            4096,
            0,
            0,
            0,
            1024,
            1,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&c, t_real as u64 * 4096 * 4)?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let want = read_f32(&golden.join("torch_lin1.bin"));
    diff_report("ff1a_real", &got, &want);
    Ok(())
}

pub fn lnf32_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let g = &enc.gpu;
    let mut pad = vec![0f32; tp as usize * 1024];
    pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("subsampling_out.bin")));
    let h = g.storage("lf_h", pad.len() as u64 * 4);
    g.queue.write_buffer(&h, 0, bytemuck::cast_slice(&pad));
    let normed = g.storage("lf_n", pad.len() as u64 * 4);
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.ln_f32(&mut cp, &h, &enc.w.layers[0].ln[0], &normed, tp);
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&normed, t_real as u64 * 1024 * 4)?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let want = read_f32(&golden.join("torch_normed0.bin"));
    diff_report("ln_f32", &got, &want);
    Ok(())
}

pub fn ln_test(enc: &Encoder) -> Result<()> {
    use bark3_pro_wgpu::encoder::pad128;
    let tp = pad128(4) as u32;
    let g = &enc.gpu;
    let mk = |n: u64| g.storage("t", n * 4);
    let h = mk(tp as u64 * 512);
    let normed = mk(tp as u64 * 512);
    let mut vals = vec![0f32; tp as usize * 1024];
    for r in 0..4usize {
        for i in 0..1024 {
            vals[r * 1024 + i] = (r as f32 + 1.0) * 0.25;
        }
    }
    g.queue.write_buffer(
        &h,
        0,
        bytemuck::cast_slice(&bark3_pro_wgpu::raw::RawTensor::f32_to_f16_words(&vals)),
    );

    let mut ctx = bark3_pro_wgpu::encoder::Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        // Burn 22 uniform slots like the real pipeline does before ln[2].
        for _ in 0..22 {
            let _ = ctx.next_slot(&[0u8; 4]);
        }
        ctx.ln(&mut cp, &h, &enc.w.layers[0].ln[2], &normed, tp);
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&normed, tp as u64 * 512 * 4)?;
    let halves: Vec<half::f16> = bytemuck::cast_slice(&bytes).to_vec();
    let values: Vec<f32> = halves.iter().map(|v| v.to_f32()).collect();
    for r in 0..6usize {
        let row = &values[r * 1024..(r + 1) * 1024];
        let nan = row.iter().filter(|v| v.is_nan()).count();
        println!("row {r}: nan={nan} first={:?}", &row[..3]);
    }
    Ok(())
}

pub fn epi_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let g = &enc.gpu;
    let mut h_pad = vec![0f32; tp as usize * 1024];
    h_pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_h_after_block.bin")));
    let h = g.storage("ep_h", tp as u64 * 512 * 4);
    g.queue.write_buffer(
        &h,
        0,
        bytemuck::cast_slice(&bark3_pro_wgpu::raw::RawTensor::f32_to_f16_words(&h_pad)),
    );
    let normed = g.storage("ep_n", tp as u64 * 512 * 4);
    let pooler = g.storage("ep_p", tp as u64 * 320 * 4);
    if std::env::var("FFN_DUMP_W").is_ok() {
        dump_ffn_weights(enc)?;
        return Ok(());
    }
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.ln(&mut cp, &h, &enc.w.layers[0].ln[4], &normed, tp);
        ctx.gemm_bias(
            &mut cp,
            &normed,
            &enc.w.proj_w,
            &enc.w.proj_b,
            &pooler,
            tp,
            640,
            1024,
            640,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = g.readback(&normed, t_real as u64 * 1024 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let want = read_f32(&golden.join("block_0_out.bin"));
    diff_report("norm_out(l0w)", &got, &want);
    let bytes = g.readback(&pooler, t_real as u64 * 640 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let want = read_f32(&golden.join("torch_pooler_l0.bin"));
    diff_report("pooler_l0", &got, &want);
    Ok(())
}

pub fn ffn_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_pro_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let g = &enc.gpu;
    let lw = &enc.w.layers[0];
    let mut n_pad = vec![0f32; tp as usize * 1024];
    n_pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_normed0.bin")));
    let normed = g.storage("fn_n", tp as u64 * 512 * 4);
    g.queue.write_buffer(
        &normed,
        0,
        bytemuck::cast_slice(&bark3_pro_wgpu::raw::RawTensor::f32_to_f16_words(&n_pad)),
    );
    let ffb = g.storage("fn_f", tp as u64 * 2048 * 4);
    let tmp = g.storage("fn_t", tp as u64 * 512 * 4);
    if std::env::var("FFN_DUMP_W").is_ok() {
        dump_ffn_weights(enc)?;
        return Ok(());
    }
    let mut ctx = Ctx::new(enc, 0);
    let mut e2 = g.device.create_command_encoder(&Default::default());
    {
        let mut cp = e2.begin_compute_pass(&Default::default());
        ctx.gemm(
            &mut cp,
            false,
            &normed,
            &lw.ff1a.buf,
            &ffb,
            tp,
            4096,
            1024,
            4096,
            0,
            0,
            0,
            1024,
            1,
        );
        if std::env::var("FFN_FRESH_A").is_ok() {
            drop(cp);
            g.queue.submit([e2.finish()]);
            g.device.poll(wgpu::PollType::wait_indefinitely())?;
            let raw = g.readback(&ffb, tp as u64 * 2048 * 4)?;
            let ffb2 = g.storage("fn_f2", tp as u64 * 2048 * 4);
            g.queue.write_buffer(&ffb2, 0, &raw);
            let mut e3 = g.device.create_command_encoder(&Default::default());
            let mut cp = e3.begin_compute_pass(&Default::default());
            ctx.silu(&mut cp, &ffb2, tp * 2048);
            ctx.gemm(
                &mut cp,
                false,
                &ffb2,
                &lw.ff1b.buf,
                &tmp,
                tp,
                1024,
                4096,
                1024,
                0,
                0,
                0,
                4096,
                1,
            );
            drop(cp);
            g.queue.submit([e3.finish()]);
            g.device.poll(wgpu::PollType::wait_indefinitely())?;
            let bytes = g.readback(&tmp, t_real as u64 * 1024 * 2)?;
            let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let want = read_f32(&golden.join("torch_ffn1_out.bin"));
            diff_report("ffn_freshA", &got, &want);
            return Ok(());
        }
        if std::env::var("FFN_DUMP_FFB").is_ok() {
            drop(cp);
            g.queue.submit([e2.finish()]);
            g.device.poll(wgpu::PollType::wait_indefinitely())?;
            let bytes = g.readback(&ffb, t_real as u64 * 4096 * 2)?;
            let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let want = read_f32(&golden.join("torch_silu_lin1.bin"));
            diff_report("silu_lin1", &got, &want);
            return Ok(());
        }
        ctx.silu(&mut cp, &ffb, tp * 2048);
        ctx.gemm(
            &mut cp,
            false,
            &ffb,
            &lw.ff1b.buf,
            &tmp,
            tp,
            1024,
            4096,
            1024,
            0,
            0,
            0,
            4096,
            1,
        );
    }
    g.queue.submit([e2.finish()]);
    g.device.poll(wgpu::PollType::wait_indefinitely())?;
    // ff2 weights are pre-scaled 0.5 -> ref = (h_ff1 - sub) directly.
    let bytes = g.readback(&tmp, t_real as u64 * 1024 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let want = read_f32(&golden.join("torch_ffn1_out.bin"));
    diff_report("ffn_half_out", &got, &want);
    Ok(())
}

fn dump_ffn_weights(enc: &Encoder) -> Result<()> {
    let layer = &enc.w.layers[0];
    let g = &enc.gpu;
    let bytes = g.readback(&layer.ff1b.buf, 1024 * 4096 * 2)?;
    let got: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let expected = std::fs::read("data/golden/ff2_l0_f32.bin")?;
    let expected: Vec<f32> = bytemuck::cast_slice(&expected).to_vec();
    let mismatches = got
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-6)
        .count();
    println!(
        "ff2 full check: {mismatches}/{} elems mismatched",
        expected.len()
    );
    println!("ff2 w[0..8]: {:?}", &got[..8.min(got.len())]);
    let bytes = g.readback(&layer.ff1a.buf, 16)?;
    let weights: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&bytes)
        .iter()
        .map(|v| v.to_f32())
        .collect();
    println!("ff1 w[0..8]: {weights:?}");
    Ok(())
}
