# SWE-bench Verified：Qwen3.5-35B-A3B TP2 单节点 H100（2P:2D）

## 范围与固定配置

- 节点：仅 `node2800`，8×H100；模型 `Qwen3.5-35B-A3B`，TP=2。
- 任务：SWE-bench Verified 500 题，结构化工具 harness；并发 64/128/192/256。
- 新跑13组：c64/c128/c192各跑当前新方法、No-reverse PD、原生
  HiCache/Mooncake；原9组完成后追加c256的全部四种方法。
- Colocated 为 4×TP2，无 P:D 比例；c64/c128/c192直接复用相同模型、
  harness和并发的既有结果，不重复占用GPU。
- c256按新方法、HiCache、No-reverse、Colocated顺序运行四组。
- 当前新方法只把拓扑从 1P:3D 改为 2P:2D；保留 `fast-tool=30s`、D→P direct lanes=4、P→D direct lanes=3、Host lanes=2、load worker=1。
- Host KV 总量统一：D→P 512 GiB、P→D 256 GiB；2P:2D 下分别为每 TP rank 128/64 GiB。
- 每组串行运行；结束后硬门禁确认 GPU、矩阵端口、Apptainer 与 `squashfuse_ll` 全部归零。失败则取消 allocation，并只在 `node2800` 重建后继续。

<!-- AUTOPILOT_RESULTS_START -->
## 自动运行进度与结果

> 最后检查：2026-10-17 15:30:12 UTC。
> 状态：`complete`；有效结果 **13/13**。
> 13组新实验均已完成。

### 核心指标

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法（2P:2D，tuned） | 500/0 | 285/500 | **2,909.24** | **1,827.85** | 5.318M |
| 64 | No-reverse PD（2P:2D） | 500/0 | 282/500 | **8,547.90** | **608.43** | 5.201M |
| 64 | 原生 HiCache/Mooncake（2P:2D） | 500/0 | 279/500 | **2,780.43** | **1,935.22** | 5.381M |
| 128 | 当前新方法（2P:2D，tuned） | 500/0 | 274/500 | **2,256.12** | **2,396.14** | 5.406M |
| 128 | No-reverse PD（2P:2D） | 500/0 | 281/500 | **8,454.83** | **625.69** | 5.290M |
| 128 | 原生 HiCache/Mooncake（2P:2D） | 500/0 | 290/500 | **2,318.97** | **2,332.66** | 5.409M |
| 192 | 当前新方法（2P:2D，tuned） | 500/0 | 265/500 | **2,339.65** | **2,340.57** | 5.476M |
| 192 | No-reverse PD（2P:2D） | 500/0 | 284/500 | **8,862.16** | **618.61** | 5.482M |
| 192 | 原生 HiCache/Mooncake（2P:2D） | 500/0 | 281/500 | **2,944.09** | **1,842.86** | 5.426M |
| 256 | 当前新方法（2P:2D，tuned） | 450/0 | 254/450 | **2,483.91** | **2,123.92** | 5.276M |
| 256 | 原生 HiCache/Mooncake（2P:2D） | 449/1 | 251/450 | **4,450.85** | **1,219.30** | 5.427M |
| 256 | No-reverse PD（2P:2D） | 450/0 | 259/450 | **8,630.84** | **634.82** | 5.479M |
| 256 | Colocated（4×TP2） | 449/1 | 255/450 | **2,659.64** | **2,120.63** | 5.640M |

### 全程与辅助指标

| 并发 | 方法 | T50 | T250 | T500 | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 当前新方法（2P:2D，tuned） | 455.66 | 1,587.65 | 3,657.94 | 0.16303 | 4,821.47 | 1,535.96 | 0 | — |
| 64 | No-reverse PD（2P:2D） | 1,303.26 | 4,945.51 | 9,314.05 | 0.05521 | 54,846.18 | 589.87 | 58 | — |
| 64 | 原生 HiCache/Mooncake（2P:2D） | 451.82 | 1,522.06 | 5,248.01 | 0.17178 | 2,760.02 | 1,074.68 | 46 | — |
| 128 | 当前新方法（2P:2D，tuned） | 567.17 | 1,433.37 | 4,964.77 | 0.23683 | 3,583.56 | 1,129.83 | 0 | — |
| 128 | No-reverse PD（2P:2D） | 2,066.08 | 5,184.66 | 10,065.47 | 0.06261 | 50,051.76 | 543.99 | 120 | — |
| 128 | 原生 HiCache/Mooncake（2P:2D） | 661.66 | 1,493.05 | 2,733.75 | 0.24135 | 10,800.04 | 2,054.32 | 97 | — |
| 192 | 当前新方法（2P:2D，tuned） | 796.84 | 1,561.25 | 4,909.12 | 0.25927 | 3,522.54 | 1,144.20 | 0 | — |
| 192 | No-reverse PD（2P:2D） | 2,168.12 | 5,437.76 | 9,423.17 | 0.05975 | 55,394.19 | 602.27 | 181 | — |
| 192 | 原生 HiCache/Mooncake（2P:2D） | 836.36 | 2,129.80 | 3,347.28 | 0.18978 | 21,009.28 | 1,684.16 | 153 | — |
| 256 | 当前新方法（2P:2D，tuned） | 843.50 | 1,698.22 | — | 0.24384 | 6,773.56 | 2,123.92 | 0 | — |
| 256 | 原生 HiCache/Mooncake（2P:2D） | 1,238.29 | 3,201.79 | — | 0.12451 | 26,592.44 | 1,219.30 | 203 | env=1, failed=1 |
| 256 | No-reverse PD（2P:2D） | 2,192.42 | 5,302.07 | — | 0.06213 | 57,075.44 | 634.82 | 241 | — |
| 256 | Colocated（4×TP2） | 1,021.66 | 1,952.54 | — | 0.24420 | 49,992.21 | 2,120.63 | 77 | env=1, failed=1 |

原始目录：

- c64 当前新方法（2P:2D，tuned）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-a-ft30-l4l3-swe500-c64-20261016-r1`
- c64 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-nr-swe500-c64-20261016-r1`
- c64 原生 HiCache/Mooncake（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-hc-swe500-c64-20261016-r1`
- c128 当前新方法（2P:2D，tuned）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-a-ft30-l4l3-swe500-c128-20261016-r1`
- c128 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-nr-swe500-c128-20261016-r1`
- c128 原生 HiCache/Mooncake（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-hc-swe500-c128-20261016-r1`
- c192 当前新方法（2P:2D，tuned）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-a-ft30-l4l3-swe500-c192-20261016-r1`
- c192 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-nr-swe500-c192-20261016-r1`
- c192 原生 HiCache/Mooncake（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-hc-swe500-c192-20261016-r1`
- c256 当前新方法（2P:2D，tuned）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-a-ft30-l4l3-swe500-c256-20261016-r3`
- c256 原生 HiCache/Mooncake（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-hc-swe500-c256-20261016-r1`
- c256 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-nr-swe500-c256-20261016-r1`
- c256 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-2p2d-col-swe500-c256-20261016-r1`

### 复用的 Colocated 对照

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 64 | Colocated（4×TP2） | 500/0 | 273/500 | **2668.89** | **2050.97** | 5.474M |
| 128 | Colocated（4×TP2） | 500/0 | 281/500 | **2206.11** | **2476.64** | 5.464M |
| 192 | Colocated（4×TP2） | 500/0 | 282/500 | **2217.44** | **2425.71** | 5.379M |

### T450窗口 P/D Forward、KV与复用

| 并发 | 方法 | P Forward/卡 | P Attention KV/引擎 | D Forward/卡 | D/共享 Attention KV/引擎 | Prefix复用率 |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法（2P:2D，tuned） | 56.06% | 3.27% | 97.07% | 27.61% | 96.49% |
| 64 | Colocated（4×TP2） | 28.17% | — | 70.57% | 11.68% | 96.09% |
| 64 | No-reverse PD（2P:2D） | 99.05% | 1.90% | 94.91% | 35.62% | 0.00% |
| 64 | 原生 HiCache/Mooncake（2P:2D） | 54.31% | 1.93% | 97.23% | 26.64% | 97.13% |
| 128 | 当前新方法（2P:2D，tuned） | 66.71% | 4.87% | 98.17% | 50.42% | 96.49% |
| 128 | Colocated（4×TP2） | 44.09% | — | 55.10% | 22.82% | 88.45% |
| 128 | No-reverse PD（2P:2D） | 98.96% | 1.90% | 95.42% | 62.12% | 0.00% |
| 128 | 原生 HiCache/Mooncake（2P:2D） | 68.08% | 3.07% | 98.59% | 46.25% | 94.12% |
| 192 | 当前新方法（2P:2D，tuned） | 66.77% | 8.44% | 98.21% | 56.37% | 96.50% |
| 192 | Colocated（4×TP2） | 53.93% | — | 44.73% | 32.39% | 82.80% |
| 192 | No-reverse PD（2P:2D） | 99.42% | 1.93% | 95.52% | 84.17% | 0.00% |
| 192 | 原生 HiCache/Mooncake（2P:2D） | 72.33% | 4.01% | 98.00% | 60.82% | 85.65% |
| 256 | 当前新方法（2P:2D，tuned） | 65.43% | 9.24% | 97.32% | 46.02% | 96.45% |
| 256 | 原生 HiCache/Mooncake（2P:2D） | 68.97% | 2.74% | 94.92% | 55.42% | 75.53% |
| 256 | No-reverse PD（2P:2D） | 99.40% | 1.94% | 95.91% | 85.92% | 0.00% |
| 256 | Colocated（4×TP2） | 60.05% | — | 38.77% | 39.07% | 73.34% |

Forward 为 workload 起点到 T450 的物理 GPU 平均 device-timer 占比；Attention KV 为同窗口内 `sglang_full_token_usage` 的时间加权均值。
Prefix 复用率为 P 侧 `prefill_cache / (prefill_cache + prefill_compute)`。
<!-- AUTOPILOT_RESULTS_END -->
## 指标口径

T450 是从 workload 测量起点到第 450 条 terminal episode 持久化的墙钟时间；Decode tok/s 使用完全相同的起止窗口读取各 Decode scheduler 的单调 token counter，并求总 token/墙钟时间。
