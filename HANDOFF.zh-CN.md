# 接手文档（中文）— bark3-wgpu

> 写给下一个 AI / 人类接手者。读完这一篇就能接着干，不用再翻聊天记录。
> 最后更新：2026-10-27（浏览器 WebGPU 已跑通；绕过所有 Tint 阻塞；low-bit 与原生仍有数值差异，见 §11）。

---

## 0. 一句话状态

- **原生完全可用、与官方 torch 逐字节一致**：门禁 `verify.ps1` ALL PASS（NVIDIA vulkan、Intel vulkan/dx12 三条路径都跑过），CI 5 job 全绿。
- **浏览器（wasm）WebGPU 已经跑通**：所有 pipeline 有效、GPU TDT 解码出结果，15 s 音频 **Plus 7.2x / Pro 9.0x 实时**（CPU 回退仍是 0.26–0.29x）。
- **但只有 bf16（Pro）与原生逐字节一致**；low-bit（Plus / 3.0 / Clean）在 Chrome（Dawn/Tint + Intel Xe）上会解出不同文本，而同一块显卡原生 D3D12/Vulkan 都对——即 **Tint 与 naga 的浮点代码生成差异**被 4-bit 权重的量化噪声放大。这是当前唯一未闭环项，见 §11。

---

## 1. 项目目标（不要偏离）

把 Bark 3.0（NVIDIA Parakeet TDT 0.6B 英文 ASR）的推理移植到 **Rust + wgpu**：

1. 与官方 PyTorch 输出**逐字节一致**（已达成，见门禁）；
2. 支持 Bark 3.0 全家族（Pro / 3.0 / Plus / Clean），**只吃官方权重**，自写容器解析；
3. **WASM 化**，给纯前端项目 `D:\EggTranslate` 用：VAD → 静音中点分段（≤60 s）→ 逐段转录 → 词级时间戳（结构与原生 JSON 一致）；
4. 已发布 https://github.com/eclipse005/bark3-wgpu（public，只有 `main`）。2026-10-27 由 `bark3-pro-wgpu` 改名：本项目支持 Bark 3.0 全系列权重，不再叫「Pro」；旧仓库名仍会自动重定向。

**硬约束（破坏即回退）**

- 不能影响原生逻辑：任何改动后 `pwsh D:/Bark/bark3-verify/verify.ps1 -Model <Pro 目录>` 必须 **ALL PASS**（golden + 3 fixture × GPU/CPU 逐字节 + cpuenc/pooldec/gputdt + JSON）。
- **只用官方权重**，不允许自造权重格式/派生 safetensors（历史包袱已清干净）。
- CLI **没有默认模型路径**：必须 `--model <dir>` 或 `BARK3_MODEL`，缺失直接报错。
- 不加 `words` 开关：纯文本给 `text`，要时间戳才给 `words`。
- ASR crate 不依赖 silero-rs，两者只通过 `cuts: number[]` 在应用层拼接。
- 提交只本地 + push 到 `origin`（SSH），不擅自合并/重置/改历史。

---

## 2. 目录与关键路径

| 作用 | 路径 |
|---|---|
| 主仓库（当前目录） | `D:\bark3-wgpu` |
| 原生二进制 | `D:\bark3-wgpu\target\release\bark3.exe` |
| 官方权重目录 | `D:\models\Underdog-Bark-3.0-{Pro,3.0,Plus,Clean}` |
| 逐字节门禁脚本 | `D:\Bark\bark3-verify\verify.ps1` + `D:\Bark\bark3-verify\data\` |
| 浏览器 Demo | `D:\bark3-demo`（`index.html` `app.js` `build.ps1` `pkg/`） |
| VAD 仓库（独立） | `D:\silero-rs`（含 `wasm/` crate `silero-vad-wasm`） |
| E2E / 探测脚本 | `D:\Bark\bark3-verify\e2e\`（`wasm_demo.py` `gpu_trace.py` `gpuinfo.py` `wgpu_probe.py`） |

权重 SHA-256 已与 HF LFS 核对：Pro `6f84b33e…e0e5`、3.0 `c18de941…`、Plus `fdd9f105…`、Clean `b67ca90d…`。

---

## 3. 实测数字（本机，供判断性能是否退化）

音频 15 s；Plus 容器（180 MB，low-bit）。

| 环境 | 权重加载 | 15 s 音频 | RTFx |
|---|---:|---:|---:|
| 原生 GPU · **NVIDIA P104-100 (Vulkan)** | 2.2–3.3 s | enc 0.106 s + dec 0.018 s | **103–105x** |
| 原生 GPU · **Intel 集显 (Vulkan, iGPU)** | 2.6 s | enc 1.512 s + dec 0.069 s | **9.5x** |
| 原生 CPU（AVX2，约 20 线程） | 1.2 s | 0.32 s | 47x |
| 浏览器 wasm · **CPU 回退（单线程、无 SIMD128）** | 21–23 s | 52–60 s | **0.26–0.29x** |
| 浏览器 wasm · **WebGPU** | 11.7 s（修复前） | 待验证 | 待验证（预期 5–9x，介于集显与原生之间） |

诊断用（原生）：

```powershell
$env:BARK3_DEVICE="intel";  .\target\release\bark3 --model D:\models\Underdog-Bark-3.0-Plus transcribe D:\Bark\bark3-verify\data\15s_16k.wav
$env:BARK3_DEVICE="nvidia"; ...   # auto / intel / nvidia / vulkan:0 / dx12:1 都可
```

输出里有一行 `[time] mel .. sub .. enc .. dec .. | audio 15.25s RTFx=..x`。

---

## 4. 构建与门禁命令

```powershell
# 原生
cargo build --release
cargo fmt --check
cargo clippy --all-targets         # 必须 0 warning
cargo test --release               # 31 passed

# 逐字节门禁（不可跳过）
pwsh D:/Bark/bark3-verify/verify.ps1 -Model D:/models/Underdog-Bark-3.0-Pro

# wasm + Demo
cargo build --release -p bark3-wasm --target wasm32-unknown-unknown
pwsh D:\bark3-demo\build.ps1       # 生成 pkg/（wasm-bindgen 0.2.129 --target web）
python -m http.server 8099 --bind 127.0.0.1 -d D:\bark3-demo

# 浏览器 E2E（真 Chrome，必须 channel="chrome"）
cd D:\Bark\bark3-verify\e2e
python wasm_demo.py cpu     # CPU 回退，约 1 分钟（加载 20 s + 转录 55 s）
python wasm_demo.py gpu     # WebGPU（当前就绪待验证）
```

crates 需走镜像：

```powershell
cargo --config 'source.crates-io.replace-with="rsproxy"' --config 'source.rsproxy.registry="sparse+https://rsproxy.cn/index/"' build
```

Git：HTTPS 不通，用 SSH：

```powershell
$env:GIT_SSH_COMMAND='ssh -i ~/.ssh/id_eclipse005 -o BatchMode=yes'
git push origin main
```

---

## 5. 未提交改动 —— 每个文件为什么改

**工作区是脏的**：下面 5.1–5.3 是上一轮（WebGPU async 化），5.4 是 2026-10-27 本轮（改名 + Tint 兼容 + 权重嗅探）。

### 5.1 CPU 在浏览器里能跑（已提交的上一个 commit `59be857` 里已含）
`src/parallel.rs`（新增）：`parallel::SPLIT` —— wasm 上为 `false`，于是 `cpu/gemm.rs`、`cpu/layer.rs`、`cpu/ops.rs`、`tdt.rs` 里那些「够大就交给 rayon」的分支全部走**旁边已有的串行分支**；`num_threads()` 在 wasm 上返回 1（同时让 split-k 退回普通 GEMM）。原生 `SPLIT = true`，行为字面不变（门禁已验）。

### 5.2 浏览器适配器请求（也已在 `59be857`）
`src/gpu.rs`：web 后端**不支持** `enumerate_adapters`，已改为 `request_adapter(HighPerformance)` → 失败再要浏览器回退适配器；`Auto` 在 wasm 上不再用 `is_real_gpu` 过滤（web 只报 `DeviceType::Other`）。

### 5.3 本轮未提交（WebGPU 真正能跑）

| 文件 | 改动 | 原因 |
|---|---|---|
| `src/browser.rs`（新） | `log()`（wasm → `console.log`；原生仅在 `BARK3_TRACE=1` 时 eprintln）+ `stage()` 计时器 | wasm 里 `eprintln!` 是黑洞，必须有面包屑才能定位卡点 |
| `src/gpu.rs` | `Gpu::wait()`（wasm 为 no-op）、`flush()` 改用它；`readback_async()`（wasm 用 `mapAsync` + JS Promise + `JsFuture` 真异步）；适配器/设备/流水线的 `log` | `device.poll(wait_indefinitely)` 在浏览器**永不返回**；`map_read()` 同步映射同样会死锁 |
| `src/tdt_gpu.rs` | `decode` → `decode_async`（原生保留同步壳 `pollster::block_on`），两处 `readback` 改 `readback_async().await` | GPU TDT 解码每 chunk 要读回结果 |
| `src/transcription.rs` | `run` → `run_async`（原生同步壳），`decode_async().await` | 同上 |
| `src/inference.rs` | 新增 `transcribe_samples_async` / `transcribe_chunked_async`；`transcribe_chunked` 拆出 `chunk_spans()` + `Span` + `ChunkAcc`（合并逻辑只写一份）；新增 `BARK3_DEVICE` 选择适配器 | 浏览器必须 await；顺带拿到集显/独显对比能力 |
| `wasm/src/lib.rs` | `transcribe` / `transcribeChunked` 变 async（JS 侧 Promise），参数改 `Vec<f32>` | 配合上面的 async |
| `src/encoder/shader_sources.rs`、`src/shaders.rs`、`src/tdt_gpu/shader.wgsl` | 所有 `bitcast<f32>(0xFF800000u)` → `-3.4028235e38` | **关键**：Chrome 的 Tint 会常量折叠出 `-inf`，然后报 `value -inf cannot be represented as 'f32'`，该 pipeline 整个失效（`add_softmax_f32` 就是这么挂的）。naga 不报错，所以原生一直是好的。对「取 max 的累加器」而言两者在有限输入上完全等价 |
| `D:\bark3-demo\app.js` | `?backend=` 参数（默认 `cpu`）；`await bark.transcribeChunked(...)` | async 化；先用 CPU 保证 Demo 可用 |

调试面包屑**保留**（原生只在 `BARK3_TRACE=1` 时走 stderr；浏览器就是控制台）。只有 `[gpu] pipeline <label>`（每次加载 52 行）已删掉：失败时现在会单独打 `[gpu] pipeline <label> failed validation: ...`。

### 5.4 2026-10-27 本轮（浏览器真正能跑 + 改名）

| 文件 | 改动 | 原因 |
|---|---|---|
| 全仓 | `bark3-pro-wgpu` → `bark3-wgpu`（crate 名、`bark3_pro_wgpu` 标识符、README/HANDOFF 的 badge 与 URL、目录名、GitHub 仓库名） | 项目支持 Bark 3.0 全系列权重，不该叫 Pro |
| `src/encoder/shader_sources.rs`、`src/shaders.rs`、`src/tdt_gpu/shader.wgsl` | `-3.4028235e38` → `-3.4028234e38` | **Tint 不接受任何大于 `f32::MAX` 的十进制字面量**（它不做就近舍入，直接报 `cannot be represented as 'f32'`）。3.4028235e38 落在 f32::MAX 之上、四舍五入后其实等于 f32::MAX，naga 接受、Tint 拒绝 |
| `src/tdt_gpu/shader.wgsl` | 去掉全部 subgroup 内建：横向归约 `sum_row` 改走 workgroup 内存（同样顺序：`Σ(j=0..7) p[j]+p[j+8]`），joint 头的 partner 交换改 `tid ^ 16` 经 workgroup 内存；`joint_main` 的 early return 改 if 谓词；`layer0/1_main`、`project_main` 的退出判断改 `workgroupUniformLoad` 取值 | wgpu 30 的 web 后端**不映射 Chrome 的 `subgroups` 特性**（`FEATURES_MAPPING` 里没有），设备拿不到该特性，带 `enable subgroups;` 的模块直接被拒；Tint 又要求 `workgroupBarrier` 必须在可证明一致的控制流里（naga 不查）。改完后同一份 WGSL 在两边都能用 |
| `src/gpu.rs` | `on_uncaptured_error` 改走 `browser::log`；wasm 上 pipeline 校验错误用 `spawn_local` 异步 `pop` 并打日志 | 浏览器没有 stderr，原来设备级错误是黑洞——第一版 37 词的结果差点查不出来 |
| `src/weights.rs` | 容器嗅探要求 `format` 以 `bark-levels-` 开头 | safetensors 头里也有 `"format":"pt"`，旧判定（只找 `"format"`）会把 Pro 的 `model.safetensors` 送进容器解析，报 `unsupported container format ""` |
| `src/diagnostics/*.rs`、其他格式化漂移 | `cargo fmt` | 上一轮遗留的 fmt 漂移，CI 的 `cargo fmt --check` 要求干净 |

---

## 6. 接手后第一件事（10 分钟）

浏览器 WebGPU 已经通了（§11），先复跑一遍确认没退化：

```powershell
cd D:\bark3-wgpu
pwsh D:\bark3-demo\build.ps1                                  # wasm + wasm-bindgen → pkg/
python -m http.server 8099 --bind 127.0.0.1 -d D:\bark3-demo    # 另一个终端，保持运行
cd D:\Bark\bark3-verify\e2e
python gpu_diag.py gpu 240                                      # 逐行打印页面日志
python json_dump.py D:\models\Underdog-Bark-3.0-Pro\model.safetensors browser_pro.json
python repeat.py                                                # 同一实例连跑三次，查确定性
```

预期：`[asr] 41 词，5 句，约 2.4 s（6x 实时）`（Plus）；Pro 加载 ~4 s、解码 1.7 s（9x）。

**下一步的活**：low-bit 与原生还差 3 个词的 80 ms 边界（§11），要追就从 `src/encoder/gemm.rs` 的 `c = c + a*b`（DXC 可能融合成 fma，naga 不融合）入手——改 `fma()` 前先跑门禁，原生当前是逐字节对齐 torch 的。

**定位某个 WGSL 在 Chrome 里编不过（通用手法）**：把该 shader 源码抠出来写成临时 `.html`，用 Playwright 打开，`device.createShaderModule({code})` + `await module.getCompilationInfo()` 打印 messages。Tint 比 naga 严，已知差异：

- 常量折叠出 `-inf`/`+inf`/NaN → 报错（本轮踩过）；
- `workgroupBarrier()` 在**非一致控制流**里 → 报错（naga 允许），写归约时让 barrier 待在 `if` 外面；
- 需要 `enable` 的扩展（subgroups 等）在 web 上默认不可用。

---

## 7. WebGPU/浏览器侧已确认的事实（省得重踩）

- **本机支持 WebGPU**：安装版 Chrome / Edge 基线参数下 `navigator.gpu.requestAdapter()` 能拿到适配器；`Win32_VideoController` 实际是 `Intel(R) Graphics` + `NVIDIA GeForce GTX 1070`（`nvidia-smi` 报 P104-100）+ `GameViewer Virtual Display`（远程会话）。
- **Playwright 自带的 Chromium 没有 WebGPU**（无 `navigator.gpu`），必须 `channel="chrome"` 或 `"msedge"`。
- `--use-angle=d3d12` 会把 WebGL 切到 SwiftShader，看着像“没有显卡”；`--use-angle=vulkan` 能看到 P104（但那是 WebGL，不是 WebGPU）。
- Chrome 在 Windows 上**忽略 `powerPreference`**（crbug 369219127）：拿到哪块卡由系统决定。想用独显：Windows 设置 → 系统 → 显示 → 图形 → 为 `chrome.exe` 选「高性能」。等价的注册表（已验证可用，重启 Chrome 生效）：

  ```powershell
  $k = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'
  Set-ItemProperty -Path $k -Type String -Name 'C:\Program Files\Google\Chrome\Application\chrome.exe' -Value 'GpuPreference=2;'
  # 验证：python D:\Bark\bark3-verify\e2e\gpu_info.py  → vendor=nvidia / architecture=pascal
  ```

  设完之后 Chrome 的 WebGPU 适配器变成 **NVIDIA P104-100（pascal）**；删掉该值即回到「让 Windows 决定」（本机是 Intel Xe 集显）。
- 浏览器内 wgpu 的限制：`wg workgroup storage` 只有 **32768 B**（原生集显报 49152）；`maxStorageBufferBindingSize` 2047 MiB。`P_ADD_SOFTMAX_F32` 的 workgroup 用量 18432 B，能过。
- **任何 readback 都必须 async**：`mapAsync` 走 JS 事件循环，同步等 = 页面死锁（踩过两次：一次在 `device.poll`，一次在 `map_read`）。
- `Device::poll` / `pollster::block_on` 在 wasm 上不能用；`Gpu::wait()` 已改成 no-op。
- 权重解码在浏览器里是**同步阻塞**的：Plus 容器 180 MB → 约 10–12 s 里含 ANS/lzma 解码。以后可以放 OPFS/Cache Storage 缓存解码结果。

---

## 8. 后续待办（按优先级）

1. **[已完成 2026-10-27] 浏览器 WebGPU 端到端**：Pro 与原生 JSON **逐字节一致**，Plus 文本一致、3 个词边界差 1 帧（§11）。剩下的活：low-bit 那 3 处 80 ms 打平；以及把 90 s / 180 s 也纳入对拍（当前只验了 15 s）。
2. **[体验] 权重缓存**：OPFS 存解码后的张量块，二次加载从 ~11 s 降到 ~2 s。
3. **[体验] WASM SIMD128**：CPU 回退目前 0.27x，因为内核是纯标量（AVX2 路径是 x86 专用的）。加 SIMD128 版 `dot16`/GEMM 内积，预计 2–4x，能到约 1x 实时。**这是“CPU 太慢”最划算的解法**。
4. **[接入] `D:\EggTranslate`**：实现它的 `AsrTranscribeResult { words[{text,start,end}] }`，英文走本地 wasm；两个 wasm（bark3 + silero-rs）用 `cuts` 契约在应用层拼。
5. **[发布] `D:\models` 上传 ModelScope**：官方同名 repo + 官方 NOTICE，不加工权重。
6. 可选：`D:\OneAsr` 接入（TXT 只需 adapter；SRT/ASS 需要给 `Transcript` 加 `words` seam）。

---

## 9. 已知精度/时间戳特性（下游会问）

- FLEURS-en 20 条 / 466 词：Pro WER 5.82% / CER 1.59%；Plus 5.60% / 1.52%；Clean 5.60% / 2.06%；3.0 6.90% / 2.71%。
- TDT duration ∈ {0..4} 帧 × 80 ms；`t = frame×0.08 − 0.25`；实测词起点系统性偏早 −106/−119/−133 ms，词尾基本准（−10 ms）；80 ms 栅格，**不是强制对齐**。
- `transcribe_chunked` 的 `cuts` 是静音**中点**（秒），每段两侧各加 0.20 s 上下文，词按**中点**归属唯一 chunk，时间平移到整条录音时间轴。

---

## 10. 交接检查清单

- [ ] `cargo fmt --check` / `cargo clippy --all-targets`（0 warning）/ `cargo test --release`（31 passed）
- [ ] `verify.ps1` **ALL PASS**
- [ ] `cargo build -p bark3-wasm --target wasm32-unknown-unknown` 通过
- [ ] `wasm_demo.py cpu` 仍能得到 41 词、5 句、文本与原生一致
- [ ] `wasm_demo.py gpu` 拿到 `Invalid ComputePipeline` 为空、RTFx 见第 6 节预期
- [ ] 清理调试面包屑与临时文件（`data/*.wgsl`、`D:\bark3-demo\wgsl_check.html`）
- [ ] 提交信息用 `feat(wasm): ...`，push 到 `origin main`（SSH），确认 CI 5 job 绿

---

## 11. 2026-10-27 浏览器 WebGPU 实测结论

环境：Chrome 正式版（Dawn + Tint）→ **Intel Xe-LPG 集显**（Chrome 在 Windows 上忽略 `powerPreference`），页面 `D:\bark3-demo`，`?backend=gpu`；音频 `D:\Bark\bark3-verify\data\15s_16k.wav`（15.00 s，VAD 给 `cuts=[]`，单 chunk）。

| 模型 | 加载 | 解码 | 实时倍率 | 与原生 `bark3 transcribe` 的 JSON |
|---|---:|---:|---:|---|
| **Pro**（bf16 1.25 GB） | 4.2 s | 1.66 s | **9.0x** | **逐字节一致**（2330 字符全同） |
| **Plus**（low-bit 180 MB） | 11.5–12.5 s | 2.4 s | **6.2x** | 文本/分句/41 词/5 段全同；3 个词的边界差 **80 ms（1 帧）** |
| Plus（CPU 回退，`?backend=cpu`） | ~20 s | 52–60 s | 0.26–0.29x | 逐字节一致 |

换显卡只改速度，不改结果（90 s 音频、Plus、低比特，同一页面同一条链路）：

| Chrome 用哪块卡 | 解码 | 实时倍率 |
|---|---:|---:|
| NVIDIA P104-100（独显，`GpuPreference=2`） | 2.32 s | **38.9x** |
| Intel Xe-LPG（集显，默认） | 5.75 s | **15.6x** |

15 s 的短片测不准——固定开销（首次 dispatch、buffer 分配）占大头，独显 5.5x、集显 6.2x 反而看不出差别；90 s 才反映稳态。两种卡上 Pro 都逐字节一致、Plus 都只在同样 3 个词上差 1 帧，说明这点差异与硬件无关，是 Tint 与原生代码生成的区别。

low-bit 那 3 处差异（浏览器 vs 原生）：

```
our            end 5.43 vs 5.51
congratulations end 14.07 vs 14.15
on             start 14.07 vs 14.15
```

都是整 80 ms 一格，属 TDT duration 打平后的 tie-break，不是逻辑错误。怀疑点：`src/encoder/gemm.rs` 的 `c = c + a*b` 可能被 DXC 融合成 fma，而 naga 不融合；4-bit 权重的量化噪声把这个 ulp 级差异放大成 1 帧。

**排查时踩的坑**

- 早前一次跑出 37 词（"fifty"/"entals"）：那是 **pkg 与并行 cargo build 竞争留下的旧 wasm**，不是代码问题。`repeat.py` 连跑三次结果稳定，现在 41 词与原生一致。改 wasm 后务必 `build.ps1` 重跑再测。
- 页面里 `pageerror: unreachable` 偶尔出现（wasm trap），但结果正常产出，尚未定位；先记着。
- 想在浏览器里开 CPU 解码（`BARK3_CPU_DEC`）会**死锁**：那条路要读回 pooler，走的是同步 `readback`。要么改成 async，要么别开。

复现/对拍脚本（仓库外，`D:\Bark\bark3-verify\e2e\`）：
`gpu_diag.py`（逐行日志）、`json_dump.py`（导出浏览器 JSON 便于 diff）、`repeat.py`（确定性）、`gpu_info.py`（Chrome 用哪块卡）、`wgsl_check.py` + `D:\bark3-demo\check\`（单个 shader 在 Chrome 里编译并打 `getCompilationInfo`）。

---

## 12. 2026-10-27 设备实测：Intel Xe 集显 / 独显 / CPU（原生 wgpu）

环境：Intel(R) Graphics（Xe-LPG，驱动 101.8860，**Vulkan**，wg workgroup storage 49152 B、**maxStorageBufferBindingSize 只有 1023 MiB**）、NVIDIA P104-100（Vulkan，2047 MiB）、CPU（AVX2，约 20 线程）。音频为 `D:\Bark\bark3-verify\data\{15s,90s,180s}_16k.wav`，RTFx 取自 `[time]` 行的 mel+sub+enc+dec。`BARK3_DEVICE=intel|nvidia` / `BARK3_CPU_ENC=1` 切换。

| 设备 | 模型 | 15 s | 90 s | 180 s | 20 条语料（corpus） |
|---|---|---:|---:|---:|---:|
| **Intel Xe 集显** | Pro（bf16） | 11.9x | 11.2x | 10.6x | **8.8x** |
| **Intel Xe 集显** | Plus（low-bit） | 9.5x | 8.8x | 8.4x | **7.6x** |
| NVIDIA P104-100 | Pro | 97.7x | 103.2x | 90.1x | 85.3x |
| NVIDIA P104-100 | Plus | 80.4x | 94.0x | 88.0x | 81.1x |
| CPU（AVX2） | Pro | — | 46.9x | — | — |
| CPU（AVX2） | Plus | — | 47.7x | — | — |

精度（20 条英文语料，202 s，WER/CER 用本仓库外脚本 `eval_device.py` 的归一化：小写 + 去标点）：

| 模型 | WER | CER | Intel 集显 | NVIDIA 独显 |
|---|---:|---:|---|---|
| Pro | 4.58% | 1.41% | 20/20 假设逐字相同 | 同左 |
| Plus | 4.91% | 1.43% | 20/20 相同 | 同左 |
| Clean | 5.10% | 2.48% | 20/20 相同 | 同左 |
| 3.0 | 6.11% | 2.52% | 20/20 相同 | 同左 |

`BARK3_DEVICE=intel pwsh D:/Bark/bark3-verify/verify.ps1 -Model D:/models/Underdog-Bark-3.0-Pro` → **ALL PASS**：集显上 GPU 输出与 torch 参考逐字节一致（golden + 3 fixture + cpuenc/pooldec/gputdt + JSON），不只是「词一样」。

**结论**

1. **本机集显没有使用价值**：9–12x 比 CPU（约 47x）慢 4–5 倍，更别说独显（90–100x）。想省电/占用小可以用，但别指望它加速。
2. **集显上 Pro 比 Plus 快约 25%**：低比特 GEMM 的 nibble 解包 + 表查在算力受限的 iGPU 上比省下来的带宽更贵；独显上两者接近（Pro 仍略快）。
3. **分块自适应是安全的**：集显 `maxStorageBufferBindingSize` 只有独显的一半，编码器的 tile 选择会不同，但 20/20 假设逐字相同、门禁也逐字节通过。
4. 跨设备精度完全一致（同一权重），差异只在速度——所以「集显能不能用」的答案是：能算对，但慢。


---

## 13. 2026-10-27 WebGPU 在独显上实测（Chrome → NVIDIA P104-100）

按 §7 的注册表把 Chrome 的 GPU 偏好设成高性能后，`navigator.gpu.requestAdapter()` 拿到 `vendor=nvidia / architecture=pascal`。测试页面 `D:\bark3-demo\check\corpus.html`（一次加载模型、连续转 20 条），脚本 `corpus_run.py` + `compare_browser.py`。

| 模型 | 15 s | 90 s | 180 s | 20 条语料 | 模型加载 |
|---|---:|---:|---:|---:|---:|
| Plus（low-bit） | 9.9x | 34.7x | 46.9x | **39–43x** | 11.2–11.6 s |
| Pro（bf16） | 10.2x | 38.7x | 49.1x | **39–40x** | 3.2–3.4 s |
| 原生同卡 Plus | 80.4x | 94.0x | 88.0x | 81.1x | 2–3 s |
| 原生同卡 Pro | 97.7x | 103.2x | 90.1x | 85.3x | 3.5 s |

精度（同一批 20 条）：

| 模型 | 浏览器 WER / CER | 原生 WER / CER | 假设逐条相同 |
|---|---|---|---|
| Plus | 4.91% / 1.43% | 4.91% / 1.43% | **20/20** |
| Pro | 4.58% / 1.41% | 4.58% / 1.41% | **20/20** |

（Pro 单条 15 s 的 JSON 之前逐字节比过：完全一致；Plus 只差 3 个词的 80 ms 边界。）

**怎么读这些数字**

- 浏览器里 **15 s 那一列偏低**是因为每次开页都要重新编译管线 + 首次 dispatch：15 s 的片段要摊掉约 0.7 s 预热，180 s 的片段几乎不用摊。常驻页面（EggTranslate 那种）看 **39–49x** 才是稳态。
- 浏览器约为原生同卡的一半（40x vs 85x），开销在 WebGPU 的提交/绑定与 JS 边界，不在算力。
- 模型加载 11 s（Plus）是**容器解码**（ANS/lzma）而不是 GPU：OPFS 缓存能把二次加载降到 ~2 s（§8 待办 2）。

**顺手修掉的一个 panic**：给 wasm 的 pipeline 校验错误加异步 `pop_error_scope` 之后，控制台开始出现 `pageerror: RuntimeError: unreachable`——wgpu 内部 `Mismatched pop_error_scope call`。错误作用域是**栈**，批量创建管线时异步 pop 会乱序。现在浏览器端不压栈，直接靠 `on_uncaptured_error` 打日志；原生仍用同步作用域并报出管线名。


---

## 14. 2026-10-27 wasm 侧优化：管线预热

**问题**：Dawn 在**首次 dispatch** 时才编译 compute pipeline。页面里第一次转录要替整次加载付这笔钱——实测 20 条语料里第一条（9.9 s 音频）耗时 **1528 ms**，其余条目 140–280 ms。

**做法**：`AsrInference::warmup_async()` 用 2 s 静音把整条链路跑一遍（编码器 + GPU TDT），wasm 的 `create()` 在权重加载完成后调用它；加载时间多约 1.3 s，用户的第一段音频不再等。同时把 `[time]` 行接进浏览器控制台，才看清 `enc` 只有 14 ms/60 s chunk（host 编码很便宜）、时间全在等待 GPU。

**结果**（20 条语料，202 s 音频，Chrome + P104-100）：

| 模型 | 预热前 | 预热后 | 原生同卡 |
|---|---:|---:|---:|
| Plus | 39–43x | **55.7x** | 81.1x |
| Pro | 39–40x | **58.6x** | 85.3x |

第一条语料：1528 ms → **167 ms**；20/20 假设仍与原生逐字相同，WER/CER 不变。

**还剩什么瓶颈**

1. **加载 12.5 s（Plus）几乎全是容器解码**（ANS + 纯 Rust LZMA），不是 GPU：Pro 的 safetensors 1.25 GB 只要 3.2 s。解法是 OPFS 缓存解码后的张量（二次加载 ~2 s，§8 待办 2）。
2. **稳态 55–59x vs 原生 81–85x**：差在 Chrome 每次 dispatch/bind group 的 JS 边界与 Dawn 校验；host 编码实测只占 14 ms，所以不是 Rust 侧的开销。
3. **CPU 回退仍 0.27x**：只有 SIMD128 能救（§8 待办 3）。


---

## 15. 2026-10-27 wasm 侧优化：OPFS 权重缓存

**问题**：Plus 的 12.5 s 加载里约 10 s 是**容器解码**（ANS + 纯 Rust LZMA），GPU 上传只占一小部分；而解码结果是确定的，只有换模型才会变。

**做法**（`src/weights_cache.rs`，仅 wasm 编译）：

- `fingerprint()`：长度 + release 名 + 三段 64 KiB 采样（FNV-1a）。整文件哈希比它要省下的解码还贵。
- `record()` 包住真正的 `WeightSource`，记下每个张量**解码后**的形态（RawTensor / LowBitTensor / 没这个形态）。
- `Recorder::persist()` 把记录写成 OPFS 里的一个扁平文件（magic + 版本 + 每张量的元数据和字节 + ModelInfo + 结束标记），在 `create()` 返回后**后台**写，不挡第一次转录。
- 下次 `create()` 先 `load()`：读文件 0.2 s → 解析 0.6–0.75 s → 直接当 `WeightSource` 用。

**结果**（Plus，20 条语料，90 s 音频，同一个持久 profile）：

| | 模型加载 | 首次转录 | 20/20 假设与原生 |
|---|---:|---:|---|
| 冷（解码 + 写缓存） | 12.8 s | 150 ms | 相同 |
| 热（OPFS 命中） | **1.9 s** | 150 ms | 相同 |

缓存文件约 **397 MB/模型**（4-bit 码保持 4-bit，不展开成 f16）。`model.safetensors`（Pro）**不缓存**：它本来就 3.2 s，缓存要 1.25 GB。

**踩过的三个坑**（都已在代码里注释）

1. `getFileHandle` 必须带 `create: true`，否则 `NotFoundError`。
2. 缓存里的 payload **不是 4 字节对齐**的，`bytemuck::cast_slice` 会 panic，wasm 里表现为页面直接卡死（等着 180 s 也等不到结果）；改成 `chunks_exact` 手工组 u32/u16。
3. 写元数据时存了**元素数**、读的时候按**字节数**消费 → 报 `truncated cache file`；现在元数据一律存字节长度。

配额：缓存走浏览器的 OPFS 配额（通常约磁盘 60%），两个 low-bit 模型约 800 MB；`persist` 失败只会打日志，不影响使用。

