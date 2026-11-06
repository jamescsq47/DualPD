# SWE-bench Verified · Qwen3.8-27B TP=2 · 单节点H100 · 1P:3D矩阵

> 创建：2026-10-14。计划在唯一允许使用的 `node2700` 上串行完成12组实验；
> 每组完成并通过零残留门禁后自动启动下一组。

## 实验范围

- 模型：Qwen3.8-27B，TP=2，BF16，context 131,072。
- 数据：SWE-bench Verified固定500题，source-order，seed 2026。
- Harness：Qwen3.8-27B既有验证路径 `openenv_structured_tool`；最多64轮、每轮最多
  8,192 token，temperature 0.6、top-p 0.95、top-k 20。
- 并发：64、128、192；每档依次运行当前新方法、Colocated、No-reverse PD、
  原生HiCache/Mooncake。
- 单节点布局：`P0=GPU0-1`、`D0=GPU2-3`、`D1=GPU4-5`、
  `D2=GPU6-7`。每个TP2组均完整位于同一NUMA域；Colocated使用4个TP2完整副本。
- 当前新方法：D→P Host总量严格为512GiB（D0每rank 86GiB，D1/D2每rank
  85GiB），P→D Host总量严格为256GiB（P0每rank 128GiB），每个D保留
  131,072 tokens的共享Decode growth reserve。原生HiCache使用baseline SGLang +
  Mooncake原生机制和256GB共享segment。
- Colocated和No-reverse使用从`b1ef6ef00`隔离出的干净baseline，只补入两项既有
  Qwen3.8-27B正确性修复：fallback GDN的`a/b.contiguous()`，以及原生Decode
  preallocation对active + ping-pong Mamba slots的原子admission；不包含任何新方法、
  D→P、Host slow path或Agentic调度代码。HiCache继续使用原目录和原生机制。

### 无效试跑审计

原c128/8,192-token reserve试跑在335/500时因D0 Attention-KV增长空间耗尽而退出；
335题均为正常terminal记录，其中149题resolved、environment error为0。该试跑未达到
T450，目录 `q38-27b-h100-1n-1p3d-agentic-swe500-c128-20261014-r1` 仅保留审计，
不计入正式矩阵。

原c64 Colocated与No-reverse误用了缺少上述两项正确性修复的`pd_baseline_b1ef6`。
其221/500与58/500、吞吐及T450全部作废，只保留原始目录用于故障审计；正式表格改用
隔离修复baseline的`correctness-r3`复跑结果。正在运行的HiCache不受此次迁移影响。

## 核心指标口径

- **T450**：从workload测量起点到第450个terminal result的墙钟时间。
- **T450前Decode吞吐**：同一时间段内，各Decode endpoint的
  `sglang_realtime_tokens_total{mode="decode"}` counter增量之和除以T450；各endpoint
  先独立做差再求和，边界位于两次scrape之间时线性插值。
- 全程吞吐包含T450后的有限任务drain和verifier长尾，仅作为辅助指标。

## 串行与清理约束

迁移时先让已在运行的c64 HiCache自然完成，再补跑c64 Colocated与No-reverse；之后
执行c128、c192的四方法序列。每组切换前必须确认GPU进程、矩阵端口、Apptainer、
`squashfuse_ll`和`faked-sysv`全部归零；未清理干净时取消并重建allocation，
不得启动下一组。

<!-- AUTOPILOT_RESULTS_START -->
## 自动运行进度与结果

> 最后检查：2026-10-16 14:19:51 UTC。
> 状态：`paused`；有效结果 **6/12**。
> 停止于：c128 No-reverse PD（1P:3D），terminal records **292/500**（正常完成292，resolved 181，environment error 0），run `q38-27b-h100-1n-1p3d-native-oneway-swe500-c128-20261014-correctness-r4`。
> 暂停原因：`Stopped by user after diagnosing c128 No-reverse prefill saturation`

### 核心指标

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法（1P:3D） | 500/0 | 252/500 | **5,382.21** | **1,877.82** | 10.107M |
| 64 | Colocated（4×TP2） | 499/1 | 245/500 | **5,753.92** | **1,738.81** | 10.005M |
| 64 | No-reverse PD（1P:3D） | 500/0 | 258/500 | **40,421.68** | **290.71** | 11.751M |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 500/0 | 252/500 | **12,780.23** | **792.56** | 10.129M |
| 128 | 当前新方法（1P:3D） | 500/0 | 256/500 | **5,380.70** | **1,999.48** | 10.759M |
| 128 | Colocated（4×TP2） | 500/0 | 248/500 | **5,796.01** | **1,733.30** | 10.046M |
| 128 | No-reverse PD（1P:3D） | 292/0（已终止） | 181/292 | — | — | — |
| 128 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — |
| 192 | 当前新方法（1P:3D） | — | — | — | — | — |
| 192 | Colocated（4×TP2） | — | — | — | — | — |
| 192 | No-reverse PD（1P:3D） | — | — | — | — | — |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — |

### 全程与辅助指标

| 并发 | 方法 | T50 | T250 | T500 | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 当前新方法（1P:3D） | 535.10 | 2,840.43 | 7,417.79 | 0.08252 | 2,213.22 | 1,450.66 | 2 | — |
| 64 | Colocated（4×TP2） | 677.37 | 3,159.35 | 7,465.45 | 0.07879 | 1,265.18 | 1,427.32 | 4 | env=1, failed=1 |
| 64 | No-reverse PD（1P:3D） | 2,622.45 | 20,435.84 | 46,012.49 | 0.01058 | 8,938.43 | 272.44 | 62 | — |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 855.49 | 6,073.75 | 14,715.88 | 0.03354 | 6,392.93 | 732.42 | 59 | — |
| 128 | 当前新方法（1P:3D） | 411.53 | 2,551.17 | 6,599.64 | 0.08050 | 2,587.43 | 1,686.58 | 5 | — |
| 128 | Colocated（4×TP2） | 457.18 | 2,811.43 | 7,210.87 | 0.07492 | 12,527.95 | 1,461.68 | 24 | — |
| 128 | No-reverse PD（1P:3D） | — | — | — | — | — | — | — | 已终止 292/500 |
| 128 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — | — | 待完成 |
| 192 | 当前新方法（1P:3D） | — | — | — | — | — | — | — | 待完成 |
| 192 | Colocated（4×TP2） | — | — | — | — | — | — | — | 待完成 |
| 192 | No-reverse PD（1P:3D） | — | — | — | — | — | — | — | 待完成 |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — | — | 待完成 |

原始目录：

- c64 当前新方法（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-agentic-swe500-c64-20261014-r2`
- c64 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-colocated4-swe500-c64-20261014-correctness-r3`
- c64 No-reverse PD（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-native-oneway-swe500-c64-20261014-correctness-r4`
- c64 原生 HiCache/Mooncake（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-hicache-mooncake-swe500-c64-20261014-r2`
- c128 当前新方法（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-agentic-swe500-c128-20261014-r2`
- c128 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-1p3d-colocated4-swe500-c128-20261014-correctness-r3`
<!-- AUTOPILOT_RESULTS_END -->

<!-- T50_T450_DETAILS_START -->
## T50–T450 P/D Forward 与吞吐

下表统一取第50条至第450条terminal result之间的主体窗口。Prefill/Decode tok/s为scheduler实际计算量。

Forward计数器包含两个TP rank，因此先按TP组求均值，再除以TP=2，得到每张物理GPU的墙钟占比。PD的P/D位于不同GPU，二者不能相加；Colocated的Extend/Decode位于同一批GPU，可以相加理解计算时间构成。

| 并发 | 方法 | 窗口(s) | Agent/s | P compute tok/s | D tok/s | P/Extend Forward/GPU | D/Decode Forward/GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | 当前新方法（1P:3D） | 4,847.10 | 0.08252 | 2,747.28 | 1,799.45 | 45.52% | 99.85% |
| 64 | Colocated（4×TP2） | 5,076.55 | 0.07879 | 1,505.02 | 1,664.68 | 9.72% | 90.06% |
| 64 | No-reverse PD（1P:3D） | 37,799.23 | 0.01058 | 9,086.06 | 267.71 | 99.99% | 87.57% |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 11,924.74 | 0.03354 | 7,398.61 | 743.99 | 92.14% | 97.52% |
| 128 | 当前新方法（1P:3D） | 4,969.17 | 0.08050 | 2,751.69 | 1,803.30 | 48.39% | 99.77% |
| 128 | Colocated（4×TP2） | 5,338.83 | 0.07492 | 16,502.71 | 1,551.45 | 50.16% | 49.74% |
| 128 | No-reverse PD（1P:3D） | — | — | — | — | — | — |
| 128 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — |
| 192 | 当前新方法（1P:3D） | — | — | — | — | — | — |
| 192 | Colocated（4×TP2） | — | — | — | — | — | — |
| 192 | No-reverse PD（1P:3D） | — | — | — | — | — | — |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — |

## T50–T450 复用率与缓存/队列

有效复用率按 `P cache / (P cache + P compute)` 计算，等价于 `1 - 实际Prefill计算量 / 逻辑Prompt量`。No-reverse关闭复用且不导出 `prefill_cache` counter，因此按精确的0%记录。

| 并发 | 方法 | 有效复用率 | P queue | D running | D prealloc / transfer | Attention KV P/D | Mamba P/D |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | 当前新方法（1P:3D） | 95.14% | 0.00 | 17.08 | 0.00 / 0.00 | 10.87% / 56.60% | 8.17% / 31.67% |
| 64 | Colocated（4×TP2） | 97.24% | 0.01 | 14.11 | 0.00 / 0.00 | 47.99%（共享） | 34.06%（共享） |
| 64 | No-reverse PD（1P:3D） | 0.00% | 52.50 | 2.30 | 0.70 / 17.98 | 6.49% / 82.15% | 0.62% / 25.33% |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 68.60% | 35.07 | 6.61 | 0.70 / 12.76 | 9.56% / 75.47% | 4.33% / 24.20% |
| 128 | 当前新方法（1P:3D） | 95.58% | 0.01 | 17.87 | 0.00 / 0.00 | 20.02% / 67.00% | 13.41% / 33.28% |
| 128 | Colocated（4×TP2） | 68.37% | 2.05 | 25.98 | 0.00 / 0.00 | 89.52%（共享） | 62.82%（共享） |
| 128 | No-reverse PD（1P:3D） | — | — | — | — | — | — |
| 128 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — |
| 192 | 当前新方法（1P:3D） | — | — | — | — | — | — |
| 192 | Colocated（4×TP2） | — | — | — | — | — | — |
| 192 | No-reverse PD（1P:3D） | — | — | — | — | — | — |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — |

Attention KV/Mamba为各TP组活跃、不可驱逐缓存池的时间均值，不是整卡HBM占比；queue、running、prealloc、transfer均为每个TP组的时间均值。空白行表示该run尚未达到T450或时序metrics尚不完整。
<!-- T50_T450_DETAILS_END -->
