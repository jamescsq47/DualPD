# SWE-bench Verified · Qwen3.5-122B-A10B TP=8 · 双节点四方法对比

本文统一汇总 c64、c128、c192、c256 四档并发和四种方法，共 16 组实验。每组都
严格使用两个 H100 节点（16×H100），不借用第三个节点承载 Router、Mooncake、
workload 或 SWE-bench verifier。

所有实验使用 SWE-bench Verified 500 题、source-order、BF16、EP=1、context
131,072、Prefill chunk/batch 8,192、`mem_fraction_static=0.8`、temperature=0.6、
top_p=0.95、top_k=20、单轮最多 8,192 tokens、最多 64 轮。主要指标为从 workload
测量起点到第 450 个 terminal result 的 **T450**；T450 前 Decode throughput 使用
所有 Decode endpoint 的 scheduler 单调 token counter 计算。

## 1. T450 主表

| 并发 | Agentic-PD | Colocated | 原生单向 PD（无 D→P 回传） | 原生 HiCache/Mooncake PD |
|---:|---:|---:|---:|---:|
| 64 | 4,025.51 s | **3,815.42 s** | 10,894.26 s | 5,052.70 s |
| 128 | 3,434.49 s | **2,830.24 s** | 11,462.79 s | 3,616.09 s |
| 192 | **3,817.13 s** | 4,711.84 s | 11,188.71 s | 9,720.99 s |
| 256 | **3,360.06 s** | 6,801.94 s | 11,399.05 s | 11,099.96 s |

c64/c128 下两个完整 TP8 副本的 Colocated 最快；并发升到 c192/c256 后，Colocated
和 HiCache 的缓存容量不足并产生大量重复 Prefill，Agentic-PD 成为最快方法。
Agentic-PD 的 T450 在四档并发下维持在 3.36--4.03 ks，而原生单向 PD 始终约为
10.9--11.5 ks。

## 2. 方法与缓存路径

| 方法 | 计算布局 | P→D | D→P | Host/cache 容量 |
|---|---|---|---|---|
| Agentic-PD | 1×P TP8 + 1×D TP8 | Agentic Direct/Host | Agentic Direct/Host | D→P 512 GiB；P→D 256 GiB |
| Colocated | 每节点 1×完整 TP8，共两个副本 | 不跨实例 | 不跨实例 | 实例内 GPU cache |
| 原生单向 PD | 1×P TP8 + 1×D TP8 | 原生 SGLang NIXL | 关闭，下一轮重新 Prefill | 无 D→P Host/cache |
| HiCache/Mooncake PD | 1×P TP8 + 1×D TP8 | 原生 SGLang NIXL | 原生 HiCache/Mooncake | Mooncake 512 GiB；Attention/Mamba Host 见下表 |

| 并发 | HiCache `mamba_full_memory_ratio` | Attention HiCache | Mamba HiCache | Mooncake |
|---:|---:|---:|---:|---:|
| 64 | 0.5 | 26 GiB/rank | 16 GiB/rank | 512 GiB |
| 128 | 0.5 | 26 GiB/rank | 16 GiB/rank | 512 GiB |
| 192 | 0.5 | 26 GiB/rank | 16 GiB/rank | 512 GiB |
| 256 | 0.9 | 26 GiB/rank | 20 GiB/rank | 512 GiB |

## 3. 完成质量与 T450 前 Decode 吞吐

| 并发 | 方法 | 正常完成 / 失败 | Resolved | T450 | T450 前 Decode tok/s |
|---:|---|---:|---:|---:|---:|
| 64 | Agentic-PD | 499 / 1 | **321 / 500** | 4,025.51 s | 1,344.08 |
| 64 | Colocated | 500 / 0 | 317 / 500 | **3,815.42 s** | **1,423.89** |
| 64 | 原生单向 PD | 500 / 0 | 315 / 500 | 10,894.26 s | 500.39 |
| 64 | HiCache/Mooncake PD | 500 / 0 | 311 / 500 | 5,052.70 s | 1,096.78 |
| 128 | Agentic-PD | 500 / 0 | 312 / 500 | 3,434.49 s | 1,566.64 |
| 128 | Colocated | 500 / 0 | 313 / 500 | **2,830.24 s** | **1,941.44** |
| 128 | 原生单向 PD | 500 / 0 | 316 / 500 | 11,462.79 s | 499.45 |
| 128 | HiCache/Mooncake PD | 500 / 0 | **322 / 500** | 3,616.09 s | 1,531.35 |
| 192 | Agentic-PD | 500 / 0 | 312 / 500 | **3,817.13 s** | **1,451.10** |
| 192 | Colocated | 500 / 0 | **317 / 500** | 4,711.84 s | 1,159.76 |
| 192 | 原生单向 PD | 500 / 0 | 298 / 500 | 11,188.71 s | 490.93 |
| 192 | HiCache/Mooncake PD | 500 / 0 | 310 / 500 | 9,720.99 s | 573.05 |
| 256 | Agentic-PD | 500 / 0 | 311 / 500 | **3,360.06 s** | **1,671.01** |
| 256 | Colocated | 500 / 0 | **317 / 500** | 6,801.94 s | 812.05 |
| 256 | 原生单向 PD | 500 / 0 | 313 / 500 | 11,399.05 s | 490.37 |
| 256 | HiCache/Mooncake PD | 491 / 9 | 301 / 500 | 11,099.96 s | 494.90 |

Resolved 的小幅差异来自采样轨迹，不作为性能指标。c64 Agentic-PD 的 1 个失败是
环境错误；c256 HiCache 的 9 个失败来自 verifier 容器在进程峰值时 `fork` 返回
EAGAIN。后者的 P、D 和 Router 全程健康，第 450 个 terminal result 与第 450 个正常
完成结果仅相差 10.88 秒，因此保留 T450，但不标记为零异常运行。

## 4. T500 与全程平均吞吐

T500 为 workload summary 的完整测量时长，包含 T450 后有限 workload drain 和
verifier 长尾，不宜单独用来衡量稳态服务能力。

| 并发 | 方法 | T500 | 全程 Prefill tok/s | 全程 Decode tok/s |
|---:|---|---:|---:|---:|
| 64 | Agentic-PD | 6,183.95 s | 2,833.16 | 914.63 |
| 64 | Colocated | 4,603.94 s | 3,221.12 | 1,241.97 |
| 64 | 原生单向 PD | 11,629.89 s | 44,659.95 | 497.55 |
| 64 | HiCache/Mooncake PD | 6,029.08 s | 2,582.61 | 984.08 |
| 128 | Agentic-PD | 4,646.94 s | 3,741.16 | 1,194.55 |
| 128 | Colocated | 5,539.03 s | 3,349.99 | 1,024.11 |
| 128 | 原生单向 PD | 13,445.40 s | 39,416.43 | 436.97 |
| 128 | HiCache/Mooncake PD | 4,302.00 s | 7,600.28 | 1,328.01 |
| 192 | Agentic-PD | 4,240.96 s | 4,345.30 | 1,420.98 |
| 192 | Colocated | 7,511.75 s | 29,636.50 | 755.26 |
| 192 | 原生单向 PD | 11,626.74 s | 42,999.40 | 489.11 |
| 192 | HiCache/Mooncake PD | 10,162.17 s | 27,260.27 | 572.08 |
| 256 | Agentic-PD | 4,913.12 s | 3,564.07 | 1,153.48 |
| 256 | Colocated | 7,761.33 s | 47,791.56 | 735.08 |
| 256 | 原生单向 PD | 11,784.99 s | 43,170.28 | 480.73 |
| 256 | HiCache/Mooncake PD | 13,328.07 s | 25,144.04 | 420.36 |

## 5. 全程实际计算量

实际 Prefill 和 Decode 均来自 scheduler 单调 counter，而不是逻辑 Prompt 长度。
这里的有效 KV cache hit rate 定义为
`max(0, 1 - 实际 Prefill / 逻辑 Prompt)`。它衡量全程有多少逻辑 Prompt token
没有被重新计算，与瞬时 Prometheus cache-hit gauge 不是同一口径。原生单向 PD
偶尔会因 scheduler 页对齐和统计边界出现实际 Prefill 略高于逻辑 Prompt，此时按
0% 记。

| 并发 | 方法 | 逻辑 Prompt | 全程实际 Prefill | KV cache hit rate | 全程 Decode |
|---:|---|---:|---:|---:|---:|
| 64 | Agentic-PD | 513.18M | 17.52M | **96.59%** | 5.66M |
| 64 | Colocated | 520.23M | **14.83M** | **97.15%** | 5.72M |
| 64 | 原生单向 PD | 521.72M | 519.39M | 0.45% | 5.79M |
| 64 | HiCache/Mooncake PD | 542.35M | 15.57M | **97.13%** | 5.93M |
| 128 | Agentic-PD | 508.59M | **17.38M** | **96.58%** | 5.55M |
| 128 | Colocated | 526.86M | 18.56M | **96.48%** | 5.67M |
| 128 | 原生单向 PD | 529.33M | 529.97M | 0.00% | 5.88M |
| 128 | HiCache/Mooncake PD | 525.82M | 32.70M | 93.78% | 5.71M |
| 192 | Agentic-PD | 539.63M | **18.43M** | **96.59%** | 6.03M |
| 192 | Colocated | 522.16M | 222.62M | 57.37% | 5.67M |
| 192 | 原生单向 PD | 502.54M | 499.94M | 0.52% | 5.69M |
| 192 | HiCache/Mooncake PD | 526.21M | 277.02M | 47.35% | 5.81M |
| 256 | Agentic-PD | 510.38M | **17.51M** | **96.57%** | 5.67M |
| 256 | Colocated | 526.91M | 370.93M | 29.60% | 5.71M |
| 256 | 原生单向 PD | 510.81M | 508.76M | 0.40% | 5.67M |
| 256 | HiCache/Mooncake PD | 502.63M | 335.12M | 33.33% | 5.60M |

| 并发 | Agentic-PD hit rate | Colocated hit rate | 单向 PD hit rate | HiCache hit rate |
|---:|---:|---:|---:|---:|
| 64 | 96.59% | **97.15%** | 0.45% | 97.13% |
| 128 | **96.58%** | 96.48% | 0.00% | 93.78% |
| 192 | **96.59%** | 57.37% | 0.52% | 47.35% |
| 256 | **96.57%** | 29.60% | 0.40% | 33.33% |

Agentic-PD 在四档并发下始终只计算约 17--18M Prefill token，说明 D→P 512 GiB
与 P→D 256 GiB Host 能稳定保留跨工具轮次状态。原生单向 PD 没有 D→P 回传，
每轮基本重新计算完整历史，因此始终执行约 500M Prefill token。Colocated 和 HiCache
在 c64/c128 仍能有效复用缓存，但从 c192 开始因容量压力和驱逐产生大量重复 Prefill。

## 6. 并发扩展摘要

| 指标 | c64 | c128 | c192 | c256 |
|---|---:|---:|---:|---:|
| 最快方法 | Colocated | Colocated | Agentic-PD | Agentic-PD |
| 最快 T450 | 3,815.42 s | 2,830.24 s | 3,817.13 s | 3,360.06 s |
| Agentic 相对 Colocated | 慢 5.5% | 慢 21.3% | **快 19.0%** | **快 50.6%** |
| Agentic 相对单向 PD | **快 63.0%** | **快 70.0%** | **快 65.9%** | **快 70.5%** |
| Agentic 相对 HiCache | **快 20.3%** | **快 5.0%** | **快 60.7%** | **快 69.7%** |

高并发的收益不是提前把工具返回前的 KV load 到 P。工具结果返回前仍严格禁止
D→P H2D；更高并发只是扩大了工具已经返回、可以合法进入 P 的 eligible-work 池。

## 7. 结果完整性

| 并发 | Agentic-PD | Colocated | 原生单向 PD | HiCache/Mooncake PD |
|---:|---|---|---|---|
| 64 | 499/500；1 个环境错误 | 500/500 | 500/500 | 500/500 |
| 128 | 500/500；terminal/Host 状态清零 | 500/500 | 500/500 | 500/500；Mooncake eviction 全部成功 |
| 192 | 500/500 | 500/500 | 500/500 | 500/500 |
| 256 | 500/500；terminal/Host 状态清零 | 500/500 | 500/500 | 491/500；9 个 verifier EAGAIN |

主表只保留满足严格双节点边界的结果。曾使用额外节点承载 verifier 或 Mooncake store
的诊断运行、未完成的启动尝试以及被后续正式复跑替代的探索性结果均不进入本表。

## 8. 核心结论

1. D→P KV 回传对 122B 多轮 agent 任务有决定性收益。Agentic-PD 相对无回传
   baseline 的 T450 在四档并发下缩短 63.0%--70.5%。
2. c64/c128 时，两个完整 TP8 Colocated 副本有足够 cache，整体速度最快；其中
   c128 Colocated 的 T450 为 2,830.24 秒。
3. c192/c256 时，Colocated 与 HiCache 的实际 Prefill 分别增长至 222.62/370.93M
   和 277.02/335.12M，缓存驱逐带来的重复计算使性能明显下降。
4. Agentic-PD 的 Prefill 计算量随并发基本不变，并在 c192/c256 获得最高的 T450 前
   Decode throughput；c256 达到 1,671.01 token/s。
5. 原生单向 PD 的 Prefill GPU 虽然持续高负载，但大量算力用于重算完整历史，四档
   并发的 Decode throughput 都只有约 490--500 token/s。
