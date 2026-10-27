use anyhow::Result;
use bark3_wgpu::encoder::{Encoder, Weight};

use super::{diff_report, read_f32};

pub fn attn32_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let pp = pad128(2 * t_real - 1) as u32;
    let g = &enc.gpu;
    let lw = &enc.w.layers[0];
    let scale: f32 = (128f32).powf(-0.5);
    let w512 = tp * 512;

    let mut normed_pad = vec![0f32; tp as usize * 1024];
    normed_pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_normed_sa.bin")));
    let pos_real = bark3_wgpu::encoder::pos_embed_f32(t_real);
    let mut pos_pad = vec![0f32; pp as usize * 1024];
    pos_pad[..pos_real.len()].copy_from_slice(&pos_real);
    let normed = g.storage("a3_normed", tp as u64 * 512 * 4);
    let pos = g.storage("a3_pos", pp as u64 * 512 * 4);
    g.queue.write_buffer(
        &normed,
        0,
        bytemuck::cast_slice(&bark3_wgpu::raw::RawTensor::f32_to_f16_words(&normed_pad)),
    );
    g.queue.write_buffer(
        &pos,
        0,
        bytemuck::cast_slice(&bark3_wgpu::raw::RawTensor::f32_to_f16_words(&pos_pad)),
    );

    let mk = |n: u64| g.storage("a3", n * 4);
    let qkv = mk(tp as u64 * 1536);
    let qu = mk(tp as u64 * 1024);
    let qv = mk(tp as u64 * 1024);
    let kh = mk(tp as u64 * 512);
    let vh = mk(tp as u64 * 512);
    let relk_flat = mk(pp as u64 * 512);
    let rkh = mk(pp as u64 * 512);
    let content = mk(tp as u64 * tp as u64 * 8);
    let bd = mk(tp as u64 * pp as u64 * 8);
    let attn = mk(tp as u64 * tp as u64 * 8);
    let attn_h = mk(tp as u64 * 1024);
    let attnout = mk(tp as u64 * 512);

    let mut ctx = Ctx::new(enc, 0);
    macro_rules! step {
        ($name:expr, $buf:expr, $words:expr, $run:expr) => {{
            let mut e2 = g.device.create_command_encoder(&Default::default());
            {
                let mut cp = e2.begin_compute_pass(&Default::default());
                $run(&mut cp);
            }
            g.queue.submit([e2.finish()]);
            g.device.poll(wgpu::PollType::wait_indefinitely())?;
            let bytes = g.readback($buf, $words * 4)?;
            let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            let ref_name = format!("a32_{}.bin", $name);
            if golden.join(&ref_name).exists() {
                let want = read_f32(&golden.join(&ref_name));
                diff_report($name, &got, &want);
            } else {
                println!("  {} (no ref) first={:?}", $name, &got[..3.min(got.len())]);
            }
        }};
    }

    step!("qu", &qu, tp as u64 * 1024, |cp: &mut wgpu::ComputePass| {
        ctx.gemm(
            cp,
            false,
            &normed,
            &lw.qkv.buf,
            &qkv,
            tp,
            3072,
            1024,
            3072,
            0,
            0,
            0,
            1024,
            1,
        );
        ctx.add_head_f32(cp, &qkv, &lw.bu, &lw.bv, &qu, &qv, tp, w512, 3072, 0);
    });
    step!(
        "kh_none",
        &kh,
        tp as u64 * 256,
        |cp: &mut wgpu::ComputePass| {
            ctx.f2h(cp, &qkv, &kh, tp, w512, 3072, 1024);
        }
    );
    step!(
        "rkh_none",
        &rkh,
        pp as u64 * 256,
        |cp: &mut wgpu::ComputePass| {
            ctx.gemm(
                cp,
                false,
                &pos,
                &lw.relk.buf,
                &relk_flat,
                pp,
                1024,
                1024,
                1024,
                0,
                0,
                0,
                1024,
                1,
            );
            ctx.f2h(cp, &relk_flat, &rkh, pp, pp * 512, 1024, 0);
        }
    );
    step!(
        "content",
        &content,
        tp as u64 * tp as u64 * 8,
        |cp: &mut wgpu::ComputePass| {
            ctx.gemm_f32(
                cp,
                false,
                &qu,
                &Weight::f16(kh.clone()),
                &content,
                tp,
                tp,
                128,
                tp,
                tp * 128,
                tp * 64,
                tp * tp,
                128,
                8,
            );
        }
    );
    step!(
        "bd",
        &bd,
        tp as u64 * pp as u64 * 8,
        |cp: &mut wgpu::ComputePass| {
            ctx.gemm_f32(
                cp,
                false,
                &qv,
                &Weight::f16(rkh.clone()),
                &bd,
                tp,
                pp,
                128,
                pp,
                tp * 128,
                pp * 64,
                tp * pp,
                128,
                8,
            );
        }
    );
    step!(
        "scores",
        &attn,
        tp as u64 * tp as u64 * 8,
        |cp: &mut wgpu::ComputePass| {
            ctx.rel_add_f32(cp, &content, &bd, &attn, t_real as u32, tp, pp, scale);
        }
    );
    step!(
        "probs",
        &content,
        tp as u64 * tp as u64 * 8,
        |cp: &mut wgpu::ComputePass| {
            ctx.softmax_f32(cp, &attn, &content, 8 * tp, t_real as u32, tp);
        }
    );
    step!(
        "attn_h",
        &attn_h,
        tp as u64 * 1024,
        |cp: &mut wgpu::ComputePass| {
            ctx.gemm_f32(
                cp,
                true,
                &content,
                &Weight::f16(vh.clone()),
                &attn_h,
                tp,
                128,
                tp,
                128,
                tp * tp,
                tp * 64,
                tp * 128,
                tp,
                8,
            );
        }
    );
    step!(
        "attnout_none",
        &attnout,
        tp as u64 * 256,
        |cp: &mut wgpu::ComputePass| {
            ctx.h2f_f32(cp, &attn_h, &attnout, tp, w512);
        }
    );
    Ok(())
}

pub fn attn_test(enc: &Encoder, golden: &std::path::Path) -> Result<()> {
    use bark3_wgpu::encoder::{Ctx, pad128};
    let t_real = 191usize;
    let tp = pad128(t_real) as u32;
    let pp = pad128(2 * t_real - 1) as u32;
    let g = &enc.gpu;
    let lw = &enc.w.layers[0];
    let scale: f32 = (128f32).powf(-0.5);

    let mut normed_pad = vec![0f32; tp as usize * 1024];
    normed_pad[..t_real * 1024].copy_from_slice(&read_f32(&golden.join("torch_normed_sa.bin")));
    let pos_real = bark3_wgpu::encoder::pos_embed_f32(t_real);
    let mut pos_pad = vec![0f32; pp as usize * 1024];
    pos_pad[..pos_real.len()].copy_from_slice(&pos_real);
    let normed = g.storage("at_normed", tp as u64 * 512 * 4);
    let pos = g.storage("at_pos", pp as u64 * 512 * 4);
    g.queue.write_buffer(
        &normed,
        0,
        bytemuck::cast_slice(&bark3_wgpu::raw::RawTensor::f32_to_f16_words(&normed_pad)),
    );
    g.queue.write_buffer(
        &pos,
        0,
        bytemuck::cast_slice(&bark3_wgpu::raw::RawTensor::f32_to_f16_words(&pos_pad)),
    );

    let mk = |n: u64| g.storage("at", n * 4);
    let qkv = mk(tp as u64 * 1536);
    let qu = mk(tp as u64 * 512);
    let qv = mk(tp as u64 * 512);
    let kh = mk(tp as u64 * 512);
    let vh = mk(tp as u64 * 512);
    let relk_flat = mk(pp as u64 * 512);
    let rkh = mk(pp as u64 * 512);
    let content = mk(tp as u64 * tp as u64 * 4);
    let bd = mk(tp as u64 * pp as u64 * 4);
    let attn = mk(tp as u64 * tp as u64 * 4);
    let attn_h = mk(tp as u64 * 512);
    let attnout = mk(tp as u64 * 512);
    let w512 = tp * 512;

    let mut ctx = Ctx::new(enc, 0);
    let step = |name: &str, run: &mut dyn FnMut(&mut wgpu::ComputePass)| -> Result<()> {
        let mut e2 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e2.begin_compute_pass(&Default::default());
            run(&mut cp);
        }
        g.queue.submit([e2.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        if let Some(buf) = match name {
            "qkv" => Some(&qkv),
            "add_head" => Some(&qu),
            "f2h_k" => Some(&kh),
            "f2h_v" => Some(&vh),
            "relk" => Some(&relk_flat),
            "f2h_rk" => Some(&rkh),
            "content" => Some(&content),
            "bd" => Some(&bd),
            "rel_add" => Some(&attn),
            "softmax" => Some(&content),
            "av" => Some(&attn_h),
            "h2f" => Some(&attnout),
            _ => None,
        } {
            let words = match name {
                "qkv" => tp as u64 * 1536,
                "bd" | "f2h_rk" | "relk" => {
                    if name == "relk" {
                        pp as u64 * 512
                    } else if name == "bd" {
                        tp as u64 * pp as u64 * 4
                    } else {
                        pp as u64 * 512
                    }
                }
                "content" | "rel_add" | "softmax" => tp as u64 * tp as u64 * 4,
                _ => tp as u64 * 512,
            };
            let bytes = g.readback(buf, words * 4)?;
            let halves: Vec<half::f16> = bytemuck::cast_slice(&bytes).to_vec();
            let got: Vec<f32> = halves.iter().map(|v| v.to_f32()).collect();
            let ref_name = if ["add_head", "f2h_k", "f2h_v", "f2h_rk"].contains(&name) {
                match name {
                    "add_head" => "at_qu",
                    "f2h_k" => "at_kh",
                    "f2h_v" => "at_vh",
                    _ => "at_rkh",
                }
            } else {
                match name {
                    "qkv" => "",
                    "relk" => "",
                    "content" => "at_content",
                    "bd" => "at_bd",
                    "rel_add" => "at_scores",
                    "softmax" => "at_probs",
                    "av" => "at_attn_h",
                    "h2f" => "at_attnout",
                    _ => "",
                }
            };
            if !ref_name.is_empty() {
                let want = read_f32(&golden.join(format!("{ref_name}.bin")));
                diff_report(name, &got, &want);
            } else {
                println!("  {name:<10} (no ref) first={:?}", &got[..3.min(got.len())]);
            }
        }
        Ok(())
    };

    let split: usize = std::env::var("ATTN_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let onepass = split == usize::MAX;
    if onepass {
        let mut e2 = g.device.create_command_encoder(&Default::default());
        {
            let mut cp = e2.begin_compute_pass(&Default::default());
            ctx.gemm(
                &mut cp,
                false,
                &normed,
                &lw.qkv.buf,
                &qkv,
                tp,
                3072,
                1024,
                3072,
                0,
                0,
                0,
                1024,
                1,
            );
            ctx.add_head(&mut cp, &qkv, &lw.bu, &lw.bv, &qu, &qv, tp, w512, 3072, 0);
            ctx.f2h(&mut cp, &qkv, &kh, tp, w512, 3072, 1024);
            ctx.f2h(&mut cp, &qkv, &vh, tp, w512, 3072, 2048);
            ctx.gemm(
                &mut cp,
                false,
                &pos,
                &lw.relk.buf,
                &relk_flat,
                pp,
                1024,
                1024,
                1024,
                0,
                0,
                0,
                1024,
                1,
            );
            ctx.f2h(&mut cp, &relk_flat, &rkh, pp, pp * 512, 1024, 0);
            ctx.gemm(
                &mut cp,
                false,
                &qu,
                &kh,
                &content,
                tp,
                tp,
                128,
                tp,
                tp * 64,
                tp * 64,
                tp * tp,
                128,
                8,
            );
            ctx.gemm(
                &mut cp,
                false,
                &qv,
                &rkh,
                &bd,
                tp,
                pp,
                128,
                pp,
                tp * 64,
                pp * 64,
                tp * pp,
                128,
                8,
            );
            ctx.rel_add(&mut cp, &content, &bd, &attn, t_real as u32, tp, pp, scale);
            ctx.softmax(&mut cp, &attn, &content, 8 * tp, t_real as u32, tp);
            ctx.gemm(
                &mut cp,
                true,
                &content,
                &vh,
                &attn_h,
                tp,
                128,
                tp,
                128,
                tp * tp / 2,
                tp * 64,
                tp * 128,
                tp,
                8,
            );
            ctx.h2f(&mut cp, &attn_h, &attnout, tp, w512);
        }
        g.queue.submit([e2.finish()]);
        g.device.poll(wgpu::PollType::wait_indefinitely())?;
        let bytes = g.readback(&attnout, tp as u64 * 512 * 4)?;
        let halves: Vec<half::f16> = bytemuck::cast_slice(&bytes).to_vec();
        let got: Vec<f32> = halves.iter().map(|v| v.to_f32()).collect();
        let want = read_f32(&golden.join("at_attnout.bin"));
        diff_report("onepass", &got, &want);
        return Ok(());
    }

    step("qkv", &mut |cp| {
        ctx.gemm(
            cp,
            false,
            &normed,
            &lw.qkv.buf,
            &qkv,
            tp,
            3072,
            1024,
            3072,
            0,
            0,
            0,
            1024,
            1,
        )
    })?;
    step("add_head", &mut |cp| {
        ctx.add_head(cp, &qkv, &lw.bu, &lw.bv, &qu, &qv, tp, w512, 3072, 0)
    })?;
    step("f2h_k", &mut |cp| {
        ctx.f2h(cp, &qkv, &kh, tp, w512, 3072, 1024)
    })?;
    step("f2h_v", &mut |cp| {
        ctx.f2h(cp, &qkv, &vh, tp, w512, 3072, 2048)
    })?;
    step("relk", &mut |cp| {
        ctx.gemm(
            cp,
            false,
            &pos,
            &lw.relk.buf,
            &relk_flat,
            pp,
            1024,
            1024,
            1024,
            0,
            0,
            0,
            1024,
            1,
        )
    })?;
    step("f2h_rk", &mut |cp| {
        ctx.f2h(cp, &relk_flat, &rkh, pp, pp * 512, 1024, 0)
    })?;
    step("content", &mut |cp| {
        ctx.gemm(
            cp,
            false,
            &qu,
            &kh,
            &content,
            tp,
            tp,
            128,
            tp,
            tp * 64,
            tp * 64,
            tp * tp,
            128,
            8,
        )
    })?;
    step("bd", &mut |cp| {
        ctx.gemm(
            cp,
            false,
            &qv,
            &rkh,
            &bd,
            tp,
            pp,
            128,
            pp,
            tp * 64,
            pp * 64,
            tp * pp,
            128,
            8,
        )
    })?;
    step("rel_add", &mut |cp| {
        ctx.rel_add(cp, &content, &bd, &attn, t_real as u32, tp, pp, scale)
    })?;
    step("softmax", &mut |cp| {
        ctx.softmax(cp, &attn, &content, 8 * tp, t_real as u32, tp)
    })?;
    step("av", &mut |cp| {
        ctx.gemm(
            cp,
            true,
            &content,
            &vh,
            &attn_h,
            tp,
            128,
            tp,
            128,
            tp * tp / 2,
            tp * 64,
            tp * 128,
            tp,
            8,
        )
    })?;
    step("h2f", &mut |cp| ctx.h2f(cp, &attn_h, &attnout, tp, w512))?;
    Ok(())
}
