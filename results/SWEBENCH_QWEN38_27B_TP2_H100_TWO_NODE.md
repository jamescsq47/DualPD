# SWE-bench Verified · Qwen3.8-27B · 双节点 H100 三方法对比

本文只比较三种方法的核心结果：完成时间、T50–T450 主体稳态吞吐，以及实际处理的
Prefill/Decode token。三组均使用同一批 SWE-bench Verified 500 题、source-order、
c128、temperature=0.6、top_p=0.95、top_k=20、最多 64 轮，并使用同一组
node1700 + node1701（共 16×H100）。

- **Agentic-PD**：4P:4D、TP=2，允许 D→P/P→D KV 快慢路径。
- **Colocated**：两个节点共 8 个完整 TP=2 实例，Prefill 和 Decode 不分离。
- **原生单向 PD（无回传）**：4P:4D、TP=2。P 完成 Prefill 后仍按原生 SGLang PD
  机制通过 NIXL 把 KV 传给 D；只关闭 D→P KV 回传，因此下一轮需要由 P 重新 Prefill
  完整历史。它不是关闭双向 KV 传输。

| KV 路径 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---|---|---|
| P→D | Agentic Direct/Host | 不跨实例传输 | **原生 SGLang NIXL** |
| D→P | Agentic Direct/Host | 不跨实例传输 | **完全关闭** |
| 原生 HiCache/Mooncake | 关闭 | 不适用 | 关闭 |

## 1. 完成时间

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| 完成 / 失败 | 499 / 1 | 500 / 0 | 500 / 0 |
| T50 | **365.10 s** | 418.58 s | 490.67 s |
| T250 | 1,584.94 s | **1,562.80 s** | 2,792.30 s |
| T450 | **2,869.92 s** | 2,971.95 s | 5,647.70 s |
| T499 | 5,404.17 s | **4,844.47 s** | 6,942.47 s |
| T500 / 全量 wall | 6,483.24 s | **4,884.46 s** | 7,466.72 s |
| T50–T450 用时 | **2,504.82 s** | 2,553.37 s | 5,157.03 s |

主要看 T450 和 T50–T450：

- Agentic-PD 的 T450 比原生单向 PD 缩短 **49.2%**；主体完成吞吐为其 **2.06×**。
- Agentic-PD 和 Colocated 的主体完成速度接近，Agentic-PD 快约 **1.9%**。
- T500 会被少数超长 verifier 主导，因此不用于判断模型服务稳态性能。

## 2. T50–T450 主体稳态吞吐

该窗口固定覆盖中间 400 个完成任务，避开冷启动和有限 workload 的尾部 drain。

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| Completion throughput | **0.15969 agent/s** | 0.15666 agent/s | 0.07756 agent/s |
| 实际 Prefill compute | 7,923.90 token/s | 4,194.21 token/s | **54,735.41 token/s** |
| Decode throughput | **3,264.67 token/s** | 3,136.41 token/s | 1,620.48 token/s |
| 窗口内实际 Prefill | 19.85M tokens | **10.71M tokens** | 282.27M tokens |
| 窗口内 Decode | 8.18M tokens | 8.01M tokens | 8.36M tokens |
| Prefill Forward / GPU | 30.66% | **9.88%** | 98.91% |
| Decode Forward / GPU | 99.81% | 89.94% | **99.90%** |
| Prefill queue mean | 0.105 | 0.133 | **48.97** |
| Decode/Serving running mean | 97.89 | **100.95** | 46.56 |

原生单向 PD 在大致相同的 Decode token 总量下，执行了 **282.27M** Prefill token；
Agentic-PD 只执行 **19.85M**，少约 **92.97%**。无回传 baseline 的 P/D GPU 同时接近
满载，P 侧平均还有 48.97 个请求排队，但 Decode throughput 只有 Agentic-PD 的
49.6%。这就是不回传 KV、每轮重算完整历史的主要代价。

Colocated 的实际 Prefill 最少，因为同一实例上的 radix cache 可以直接复用历史；
Agentic-PD 的 Prefill 工作量略高，但它把四个 D 组组成更大的 Decode batch，因此主体
Decode throughput 略高于 Colocated。

## 3. 数据与 token 特性

这里区分三个概念：

- **逻辑 Prompt**：每次模型调用看到的完整上下文长度，历史 token 会在后续轮次重复计入。
- **实际 Prefill**：GPU scheduler 实际执行的 Prefill compute token；复用 KV 后会远小于
  逻辑 Prompt。
- **Decode**：模型实际生成的 token。

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| SWE-bench resolved | **247 / 500（49.4%）** | 238 / 500（47.6%） | 246 / 500（49.2%） |
| 模型调用轮数总计 | 15,550 | 15,126 | **15,764** |
| 每题平均轮数 | 31.10 | 30.25 | **31.53** |
| 逻辑 Prompt 总量 | 334.82M | 310.40M | 333.93M |
| GPU 实际 Prefill 总量 | 25.08M | **14.49M** | 334.25M |
| 实际 Prefill / 逻辑 Prompt | 7.49% | **4.67%** | 100.09% |
| Decode 总量 | **10.58M** | 10.30M | 10.50M |
| 每轮逻辑 Prompt | 21,532 tokens | 20,521 tokens | 21,183 tokens |
| 每轮实际 Prefill | 1,613 tokens | **958 tokens** | 21,203 tokens |
| 每轮 Decode | 680 tokens | **681 tokens** | 666 tokens |

三组的逻辑 Prompt、Decode 总量、轮数和 resolved rate 都在相近范围，因此对比没有被明显
不同的轨迹长度所主导。最关键的区别是实际 Prefill：

- Agentic-PD 通过 KV 回传，把约 **92.5%** 的逻辑 Prompt 从重复 Prefill 中消除。
- Colocated 通过本地 radix cache 消除约 **95.3%**。
- 原生单向 PD 没有 D→P KV，实际 Prefill 基本等于完整逻辑 Prompt。

原生单向 PD 的 100.09% 略高于 100%，来自 scheduler token counter 的页对齐和统计边界，
不是生成了额外的用户 Prompt。

## 4. 结论

1. **无 KV 回传会让 PD 性能下降约一半。** Agentic-PD 的 T50–T450 completion
   throughput 是原生单向 PD 的 2.06×，Decode throughput 是 2.01×。
2. **瓶颈是重复 Prefill。** 原生单向 PD 在主体窗口执行 282.27M Prefill token，约为
   Agentic-PD 的 14.22×，P Forward 达 98.91% 且长期排队。
3. **Agentic-PD 与 Colocated 主体速度接近。** Agentic-PD 的 completion throughput
   高 1.9%，Decode throughput 高 4.1%；单次实验不足以把约 2% 的差异解释为稳定收益。
4. **三组数据特性可比。** 每题约 30–32 轮、每轮约 20–22K 逻辑 Prompt、约 666–681
   Decode token，resolved rate 为 47.6%–49.4%。

## 5. 原始结果

- Agentic-PD：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-4p4d-swe500-c128-20261002-r1`
- Colocated：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-colocated-latest-swe500-c128-20261002-r2`
- 原生单向 PD：
  `/home/shangy/dualpd/slime/examples/pd/runs-host/qwen38_27b_native_pd_tp2_4p4d_swe500_c128_correctnessfix_20261002`

原生单向 PD 使用干净的 `pd_baseline` 分支，仅加入 Qwen3.5 GDN tensor contiguous
正确性修复和 Mamba slot admission 修复；没有加入 D→P、Host slow path 或 Agentic
调度语义。错误输出的早期诊断运行不纳入上述比较。
