# SWE-bench Verified · Qwen3.8-27B · 历史双节点实验

这份文档记录较早在 `a10+a11` 两个节点上完成的 Qwen3.8-27B 实验。有效的正式实验
只有两组：Colocated baseline 和 Agentic-PD。另一次 Colocated `r1` 为误运行，不纳入
任何结果。

两组共同配置为：SWE-bench Verified 500 题、source-order、c128、TP=2、16 张 GPU、
context 131,072、Prefill chunk/batch 8,192、最多 64 轮、temperature=0.6、top_p=0.95、
top_k=20、`mamba_full_memory_ratio=0.90`。

这批历史实验与最新 H100 三方法实验的机器、代码和 Mamba ratio 不同，不能把绝对数值
跨报告直接比较。

## 1. 做了哪些实验

| 指标 | Agentic-PD | Colocated |
|---|---|---|
| 正式 run | `qwen38-27b-tp2-global-c128-evictfix-full-20260930` | `qwen38-27b-tp2-colocated-2node-c128-r2` |
| 实例布局 | a10：4 个 P TP=2；a11：4 个 D TP=2 | 两节点共 8 个完整 TP=2 实例 |
| KV 复用 | D→P、P→D Direct/Host | 本地 Radix/Mamba cache |
| Host 容量 | D→P 512 GiB；P→D 128 GiB | 不适用 |
| 正式结果 | 500/500 完成，0 失败 | 500/500 完成，0 失败 |

## 2. 完成时间

| 指标 | Agentic-PD | Colocated |
|---|---:|---:|
| T250 | 2,531.69 s | **1,982.58 s** |
| T450 | 4,973.31 s | **3,630.61 s** |
| T500 / 全量 wall | 6,499.20 s | **5,144.26 s** |
| T250–T450 用时 | 2,441.62 s | **1,648.03 s** |
| T250–T450 completion throughput | 0.08191 agent/s | **0.12136 agent/s** |
| 全程收尾速率 | 0.0769 agent/s | **0.0972 agent/s** |

这批旧实验中，Agentic-PD 的 T450 比 Colocated 慢 **37.0%**；T250–T450 中间完成段
吞吐低 **32.5%**。这与最新 H100 实验中两者主体吞吐基本持平的结果不同，说明之后的
调度、传输和控制面修改带来了实质改善。

## 3. 中间稳态吞吐

当时保存的引擎稳态窗口并不完全相同：Colocated 使用 300–1,500 秒，Agentic-PD 使用
约 1,249–5,229 秒。因此下表适合解释资源状态，不应当作严格同窗 benchmark；严格的
任务完成对比应看上一节的 T250–T450。

| 指标 | Agentic-PD | Colocated |
|---|---:|---:|
| 实际 Prefill compute | 2,712.04 token/s | **3,936.05 token/s** |
| Decode throughput | 1,660.16 token/s | **3,018.05 token/s** |
| Decode / TP2 服务组 | **415.04 token/s** | 377.26 token/s |
| Prefill Forward / GPU | 26.56% | 13.29% |
| Decode Forward / GPU | **99.74%** | 85.86% |
| Decode running 总计 | 62.22 | **118.88** |
| Attention KV 使用率 | D 73.80%；P 16.65% | 52.04% |

Agentic-PD 的单个 D TP2 组比 Colocated 单组快约 10%，但它只有 4 个 D 组，而
Colocated 有 8 个完整实例都能 Decode；同时 Agentic-PD 的 D running 只有 62.22，
远低于 Colocated 的 118.88。因此总 Decode throughput 仍明显落后。

## 4. 数据与 token 特性

| 指标 | Agentic-PD | Colocated |
|---|---:|---:|
| SWE-bench resolved | 265 / 500（53.0%） | **269 / 500（53.8%）** |
| 每题平均模型轮数 | 32.75 | 31.80 |
| 逻辑 Prompt 总量 | 349.92M tokens | 339.68M tokens |
| 每题逻辑 Prompt | 699,840 tokens | 679,355 tokens |
| GPU 实际 Prefill 总量 | 16.72M tokens | **12.50M tokens** |
| 每题实际 Prefill | 33,449 tokens | **24,995 tokens** |
| 实际 Prefill / 逻辑 Prompt | 4.78% | **3.68%** |
| Decode 总量 | 10.31M tokens | **10.33M tokens** |
| 每题 Decode | 20,616 tokens | 20,658 tokens |

两组的轮数、逻辑 Prompt、Decode 总量和 resolved rate 很接近，轨迹具有可比性。旧版
Agentic-PD 比 Colocated 多执行 **33.8%** 实际 Prefill，但 Decode 总量几乎相同；这是
其 P 侧工作量更高、端到端完成更慢的重要原因之一。

## 5. 结论

1. 这批历史实验完成了 **Agentic-PD 和 Colocated 两种正式方法**；两组均为 500/500、
   0 失败，另有一个错误启动的 Colocated `r1` 被排除。
2. 旧版 Agentic-PD 的单 D 组效率略高，但总 D 组只有 Colocated 的一半，且 running
   不足，导致 T250–T450 completion throughput 低 32.5%。
3. 两组生成了约 10.3M Decode token，数据长度接近；性能差异不能用更短或更简单的
   Agentic-PD 轨迹解释。
4. 旧版 Agentic-PD 的实际 Prefill 比 Colocated 多 4.22M tokens（33.8%），说明当时
   KV 复用和 P/D 调度仍有明显优化空间。

## 6. 原始结果入口

- Agentic-PD：`runs/dualpd/qwen38-27b-tp2-global-c128-evictfix-full-20260930`
- Colocated：`runs/dualpd/qwen38-27b-tp2-colocated-2node-c128-r2`
- 误运行、已排除：`qwen38-27b-tp2-colocated-2node-c128-r1`

最新 H100 三方法实验见
`SWEBENCH_QWEN38_27B_TP2_H100_TWO_NODE.md`。
