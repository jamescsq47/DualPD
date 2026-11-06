# SWE-bench Verified · MiniMax-M2.7 TP8/EP8 · 双节点方法对比

> 状态：全部正式实验已完成。主指标为 T450；达到 T450 后即可停止，因此不要求 T500。

## 总览：T450 与窗口内吞吐

以下按“并发 → Agentic-PD → No-reverse PD → HiCache/Mooncake PD →
Colocated”的固定顺序排列。c64/c96 只运行了 Agentic-PD；四种方法的公平对照从
c128 开始。

| 并发 | 方法 | Resolved | **T450** | 0–T450 完成吞吐 | T50–T450 完成吞吐 | **0–T450 Decode tok/s** |
|---:|---|---:|---:|---:|---:|---:|
| 64 | Agentic-PD | 240/500 | **1,543.62 s** | 0.292 req/s | 0.288 req/s | ≈1,295† |
| 96 | Agentic-PD | 251/500 | **1,555.78 s** | 0.289 req/s | 0.288 req/s | ≈1,240† |
| 128 | Agentic-PD | 251/500 | **1,373.14 s** | 0.328 req/s | 0.335 req/s | ≈1,485† |
| 128 | No-reverse PD | 239/500 | **2,424.01 s** | 0.186 req/s | 0.201 req/s | 831 |
| 128 | HiCache/Mooncake PD | 247/500 | **1,326.54 s** | 0.339 req/s | 0.369 req/s | 1,606 |
| 128 | Colocated | 250/500 | **1,101.10 s** | 0.409 req/s | 0.429 req/s | 2,032 |
| 192 | Agentic-PD | 257/500 | **1,441.05 s** | 0.312 req/s | 0.332 req/s | 1,535 |
| 192 | No-reverse PD | 243/500 | **2,448.28 s** | 0.184 req/s | 0.195 req/s | 832 |
| 192 | HiCache/Mooncake PD | 238/500 | **1,397.32 s** | 0.322 req/s | 0.354 req/s | 1,573 |
| 192 | Colocated | 236/500 | **1,004.23 s** | 0.448 req/s | 0.443 req/s | 2,109 |
| 256 | Agentic-PD | 246/500 | **1,310.80 s** | 0.343 req/s | 0.358 req/s | 1,558 |
| 256 | No-reverse PD | 242/500 | **2,679.12 s** | 0.168 req/s | 0.184 req/s | 802 |
| 256 | HiCache/Mooncake PD | 241/500 | **1,582.34 s** | 0.284 req/s | 0.321 req/s | 1,430 |
| 256 | Colocated | 242/500 | **1,359.99 s** | 0.331 req/s | 0.359 req/s | 1,525 |

口径：T450 是 workload measurement start 到第 450 个 verifier terminal 的时间；
0–T450 完成吞吐为 `450/T450`，T50–T450 完成吞吐为
`400/(T450-T50)`。0–T450 Decode tok/s 使用 measurement start 到 T450 的
scheduler token counter；† Agentic c64/c96/c128 没有持久化 counter，只能列出
T50–T450 内 SGLang `Decode batch` 日志的区间吞吐样本均值作为近似参考，其余行均为
Prometheus 单调计数器的精确窗口差分。Resolved 是停止清理前已落盘结果在固定
SWE500 中的通过数，不参与吞吐计算。

## 计算、复用与缓存明细

| 并发 | 方法 | Prefill Forward/GPU | Decode Forward/GPU | Prompt tok/s | 新增 Prefill tok/s | P 前缀复用率 | Decode tok/s | P/共享 KV 均值 | D KV 均值 | D running 均值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | Agentic-PD | ≈97.56%† | ≈99.28%† | 67,421 | 4,006 | 94.06% | ≈1,295† | 12.47% | 33.43% | 23.06 |
| 96 | Agentic-PD | ≈92.68%† | ≈98.66%† | 70,794 | 3,969 | 94.39% | ≈1,240† | 19.69% | 36.16% | 23.39 |
| 128 | Agentic-PD | ≈95.04%† | ≈97.81%† | 76,986 | 4,280 | 94.44% | ≈1,485† | 23.86% | 45.90% | 30.72 |
| 128 | No-reverse PD | 99.69% | 99.82% | 42,177 | 42,177 | **0.00%** | 825 | 2.11% | 91.46% | 11.54 |
| 128 | HiCache/Mooncake PD | 63.17% | 99.33% | 93,580 | 5,616 | 94.00% | 1,697 | 39.33% | 83.80% | 37.92 |
| 128 | Colocated | 14.56% | 80.16% | 105,132 | 6,144 | 94.16% | 2,004 | 28.02%（共享） | — | 52.42‡ |
| 192 | Agentic-PD | 62.84% | 98.58% | 76,413 | 4,073 | 94.67% | 1,476 | 30.99% | 48.03% | 31.51 |
| 192 | No-reverse PD | 99.67% | 99.80% | 42,179 | 42,179 | **0.00%** | 806 | 2.11% | 93.59% | 11.12 |
| 192 | HiCache/Mooncake PD | 48.60% | 99.52% | 85,434 | 5,532 | 93.53% | 1,611 | 37.22% | 84.24% | 36.04 |
| 192 | Colocated | 13.71% | 61.68% | 99,375 | 5,876 | 94.09% | 1,983 | 32.13%（共享） | — | 64.78‡ |
| 256 | Agentic-PD | 65.93% | 98.43% | 79,714 | 4,814 | 93.96% | 1,573 | 36.40% | 43.89% | 32.44 |
| 256 | No-reverse PD | 98.58% | 99.53% | 39,909 | 39,909 | **0.00%** | 737 | 2.14% | 94.72% | 10.20 |
| 256 | HiCache/Mooncake PD | 52.04% | 97.64% | 79,683 | 9,675 | 87.86% | 1,461 | 20.58% | 66.83% | 33.25 |
| 256 | Colocated | 13.06% | 70.55% | 80,496 | 5,913 | 92.65% | 1,540 | 16.61%（共享） | — | 31.47‡ |

所有明细均使用 T50–T450 窗口。Prompt tok/s 包含缓存命中的输入 token；新增
Prefill tok/s 是实际送入 Prefill Forward 的 token，复用率按
`cached/(cached+new)` 计算。No-reverse 没有 D→P 回传，三组窗口内缓存命中为 0，
因此 P Forward 几乎持续满载，但 D running 仅约 10–12，D KV 却达到 91%–95%。

† Agentic c64/c96/c128 的 Forward 列是有 Forward batch 日志的 1 秒桶覆盖率，
不是 GPU device timer；c192/c256 与所有 baseline 行才是 TP8 device timer 除以
8 GPU 和窗口时长。‡ Colocated 没有固定 P/D 角色：Forward 两列分别是两个 TP8
副本合计的 Prefill/Decode Forward GPU 占比，KV 为共享池均值，running 是两个
副本之和。

## Agentic-PD 双向路径与 Host 状态

| 并发 | D→P Direct / Host restore | D→P 快/慢比例 | P→D Direct / Host restore | P→D 快/慢比例 | D→P Host entry 均值/峰值 | D→P Host GiB 均值/峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 42 / 90 | 31.82% / 68.18% | 15 / 30 | 33.33% / 66.67% | 3.36 / 14 | 11.88 / 50.35 |
| 96 | 110 / 280 | 28.21% / 71.79% | 57 / 196 | 22.53% / 77.47% | 8.60 / 34 | 26.36 / 86.60 |
| 128 | 87 / 286 | 23.32% / 76.68% | 43 / 228 | 15.87% / 84.13% | 9.95 / 31 | 29.63 / 94.42 |
| 192 | 82 / 271 | 23.23% / 76.77% | 36 / 257 | 12.29% / 87.71% | 15.22 / 43 | 46.37 / 124.80 |
| 256 | 84 / 217 | 27.91% / 72.09% | 29 / 268 | 9.76% / 90.24% | 39.32 / 119 | 81.36 / 234.07 |

路径计数只统计窗口内已完成 ownership handoff 的 generation；跨越 T50/T450
边界、已 store 但尚未 restore 的对象不计入比例。五组停止清理后均确认
D→P/P→D Host、pending、running 与 physical KV 清零。

## 固定配置与方法定义

| 项目 | 配置 |
|---|---|
| 模型 | MiniMaxAI/MiniMax-M2.7，revision `d494266a4affc0d2995ba1fa35c8481cbd84294b` |
| 权重 | 官方 block-FP8 E4M3 checkpoint；BF16 activation/non-quantized modules |
| GPU | 双节点，16×H100 80GB；每个 TP8/EP8 实例独占一台 8-GPU 节点 |
| 数据 | SWE-bench Verified 500，source-order，OpenAI tools，Apptainer verifier |
| 上下文 | 131,072；Prefill chunk/batch 8,192；单轮最多 8,192 token、最多 64 轮 |
| 采样 | temperature=0.6、top_p=0.95、top_k=20、seed=2026 |
| HBM | `mem_fraction_static=0.8`；KV cache BF16 |
| Agentic-PD | 1×P TP8/EP8 + 1×D TP8/EP8；工具窗口 1s，Direct admission+建连窗口 1s，拥堵重算关闭 |
| Agentic Host | D→P 64 GiB/rank（总 512 GiB）；P→D 32 GiB/rank（总 256 GiB） |
| No-reverse PD | 1×P + 1×D；保留原生 P→D，关闭 D→P KV 回传，多轮回到 P 时重新 Prefill |
| HiCache/Mooncake PD | 1×P + 1×D；原生 HiCache/Mooncake，`wait_complete`；Mooncake store 512 GiB |
| Colocated | 每节点 1×TP8/EP8 完整副本，共两个副本；没有固定 P/D 划分 |

MiniMax-M2.7 是纯 Attention MoE，Agentic 数据面只传 Attention KV，不创建
Qwen3.5 Mamba side-pool。模型必须使用 EP=8；TP8/EP1 与其 FP8 expert block shape
不兼容。

## 工程验证摘要

- 模型 151/151 manifest 文件完整；TP8/EP8、`minimax-m2` tool parser 与
  `minimax-append-think` reasoning parser 已完成双节点 smoke。
- 修复了纯 Attention 模型的 D→P Host 公共前缀恢复：Host 保留完整源快照，P 只恢复
  下一轮仍共享的逐层 Attention KV 前缀，不加载已失效的 decode 尾部；相关回归
  184/184 通过。
- c256 verifier 曾因每个 Apptainer SIF 的多线程 squashfuse 挂载造成 host task
  耗尽。正式结果改为单线程 squashfuse transport 后，九组 baseline 均无
  fork/EAGAIN、CUDA、OOM 或环境错误。
- 所有正式结果停止后都完成作用域清理；两节点 GPU 进程、SIF 挂载和运行残留为 0。

## 正式结果目录

Agentic-PD：

- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-agentic-tp8ep8-two-node-swe500-c64-20261015-r2`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-agentic-tp8ep8-two-node-swe500-c96-20261015-r2`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-agentic-tp8ep8-two-node-swe500-c128-20261015-r2`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-agentic-tp8ep8-two-node-swe500-c192-20261015-r2`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-agentic-tp8ep8-two-node-swe500-c256-20261015-r2`

Baseline：

- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-native-oneway-1p1d-swe500-c128-20261016-r4`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c128-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-colocated-2replica-swe500-c128-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-native-oneway-1p1d-swe500-c192-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c192-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-colocated-2replica-swe500-c192-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-native-oneway-1p1d-swe500-c256-20261016-r6`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c256-20261016-r11`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-colocated-2replica-swe500-c256-20261016-r6`

<!-- MINIMAX_M27_HIGH_CONCURRENCY_START -->
## c320/c384 自动运行状态

> 最后检查：2026-10-17 06:36:54 UTC；状态：`paused`；正式结果 **6/8**。
> 当前：c384 HiCache/Mooncake PD，episode progress **24/500**，run `minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c384-20261017-r1`。
> 暂停原因：`RuntimeError: verifier spawn/fork errors=10`

| 并发 | 方法 | 完成/失败 | Resolved | **T450** | 0–T450 Decode tok/s | 异常 |
|---:|---|---:|---:|---:|---:|---|
| 320 | Agentic-PD | 476/0 | 234/500 | 1,457.76 s | 1,474.36 | env=0, spawn=0 |
| 320 | No-reverse PD | 459/0 | 226/500 | 2,541.06 s | 844.00 | env=0, spawn=0 |
| 320 | HiCache/Mooncake PD | 479/0 | 234/500 | 1,476.28 s | 1,478.20 | env=0, spawn=0 |
| 320 | Colocated | 481/0 | 232/500 | 1,276.46 s | 1,705.13 | env=0, spawn=0 |
| 384 | Agentic-PD | 484/0 | 240/500 | 1,448.11 s | 1,402.86 | env=0, spawn=0 |
| 384 | No-reverse PD | 460/0 | 226/500 | 2,350.26 s | 872.03 | env=0, spawn=0 |
<!-- MINIMAX_M27_HIGH_CONCURRENCY_END -->

<!-- MINIMAX_M27_C384_C448_START -->
## c384 修复重跑与 c448 自动实验

> 最后检查：2026-10-17 09:48:13 UTC；状态：`complete`；正式结果 **4/4**。
> c384 HiCache 修复重跑及三组 c448 实验均已完成。

| 并发 | 方法 | 完成/失败 | Resolved | **T450** | 0–T450 Decode tok/s | 异常 |
|---:|---|---:|---:|---:|---:|---|
| 384 | HiCache/Mooncake PD | 493/0 | 257/500 | 1,654.43 s | 1,346.04 | env=0, spawn=0 |
| 448 | Agentic-PD | 497/0 | 255/500 | 1,428.60 s | 1,455.13 | env=0, spawn=0 |
| 448 | HiCache/Mooncake PD | 498/0 | 236/500 | 1,483.18 s | 1,387.35 | env=0, spawn=0 |
| 448 | No-reverse PD | 454/0 | 227/500 | 2,395.94 s | 809.66 | env=0, spawn=0 |

同一 c448 配置下，Agentic-PD 相对 HiCache/Mooncake 的 T450 缩短
**54.58 秒（3.68%）**，0–T450 Decode 吞吐提高 **4.89%**；相对
No-reverse PD 的 T450 缩短 **967.34 秒（40.37%）**。

P 侧固定 shape 的 Forward/Host H2D 分解见
[`MINIMAX_M27_C448_P_FORWARD_MICROBENCH.md`](MINIMAX_M27_C448_P_FORWARD_MICROBENCH.md)。

稳态计算与缓存指标统一取 T50–T450：

| 并发 | 方法 | P Forward/GPU | D Forward/GPU | Prompt tok/s | 新增 Prefill tok/s | P 复用率 | Decode tok/s | P KV 均值 | D KV 均值 | D running 均值 | P queue 均值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | HiCache/Mooncake PD | 63.70% | 96.96% | 75,414 | 19,368 | 74.32% | 1,330 | 15.06% | 76.35% | 28.33 | 27.37 |
| 448 | Agentic-PD | 63.68% | 98.21% | 78,458 | 4,340 | **94.47%** | **1,462** | 26.07% | 41.31% | 28.98 | 0.41 |
| 448 | HiCache/Mooncake PD | 64.54% | 98.78% | 79,917 | 17,807 | 77.72% | 1,401 | 17.65% | 73.41% | 28.77 | 24.50 |
| 448 | No-reverse PD | 99.64% | 99.79% | 42,178 | 42,178 | 0.00% | 746 | 2.09% | 95.00% | 10.14 | 84.58 |

高并发 harness 修复包括：宿主 Apptainer Go 控制进程使用
`GOMAXPROCS=1`，容器内命令按 `apptainer_cpu=2` 设置
`GOMAXPROCS=2`；每个 T450 run 停止后按精确 run-id 卸载未完成任务留下的
FUSE rootfs。修复后的 c384/c448 四组均无 verifier fork/EAGAIN、环境、CUDA
或 OOM 错误；最终两节点 GPU 进程和 MiniMax FUSE mount 均为 0。
<!-- MINIMAX_M27_C384_C448_END -->
