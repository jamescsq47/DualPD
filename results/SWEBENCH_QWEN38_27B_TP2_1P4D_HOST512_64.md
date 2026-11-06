# SWE-bench Verified · Qwen3.8-27B · TP2 · 双节点 10×H100

本文比较同一批 SWE-bench Verified 500 题、同一总硬件预算下的三种部署：
Agentic-PD 1P:4D、5 个 Colocated 实例、原生单向 PD 1P:4D。三组均为
source-order、c128、`openai_tools`，采样参数为 temperature=0.6、top_p=0.95、
top_k=20；单轮最多 8192 tokens、最多 64 轮。模型为 Qwen3.8-27B BF16，
TP=2、EP=1，context=131072，prefill chunk/batch=8192，
`mem_fraction_static=0.8`、`mamba_full_memory_ratio=0.5`。

## 1. 方法与运行有效性

| 配置 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---|---|---|
| 实例布局 | node1700：1×P TP2；node1701：4×D TP2 | 两节点共 5×TP2 完整实例 | node1700：1×P TP2；node1701：4×D TP2 |
| GPU 数 | 2P + 8D，共 10×H100 | 共 10×H100 | 2P + 8D，共 10×H100 |
| P→D | Agentic Direct/Host | 不适用 | 原生 SGLang NIXL |
| D→P | Direct，1 s 失败后转 Host | 本地前缀缓存 | 关闭；下一轮在 P 完整重算 |
| Host 容量 | D→P 64 GiB/rank，共 512 GiB；P→D 32 GiB/rank，共 64 GiB | 不适用 | 不适用 |
| 工具/下一轮窗口 | 1.0 s | 不适用 | 不适用 |
| 正式运行状态 | **500 completed / 0 failed** | **500 / 0** | **499 / 1** |
| 服务侧 fatal | 0 | 0 | 0 |
| 结果有效性说明 | terminal cleanup 全部通过 | 使用与 Agentic-PD 相同的 `pd` SGLang；Agentic KV 回传关闭 | 1 条 agent 工具超时被记为 harness infrastructure failure |

原生单向 PD 的失败题为 `sphinx-doc__sphinx-9320`：模型生成的复现命令进入无限
交互循环；600 秒工具超时后，Apptainer broker 未及时返回标准超时码。该题在第 354 个
terminal record 结束，未被 verifier 评分。此问题不属于模型服务 fatal；实验后已给 broker
补上有界超时归一化，定向测试 3/3 通过。本文仍明确按 **499/1** 报告，不把该题伪装成
正常完成。

## 2. 完成时间

计时从第一题开始到相应数量的 terminal record 结束，不含模型启动和预注册。

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| T50 | **420.04 s** | **447.41 s** | **1,165.27 s** |
| T250 | **1,688.87 s** | **1,777.72 s** | **9,712.94 s** |
| T450 | **3,115.30 s** | **3,361.86 s** | **20,811.04 s** |
| T500 / 全量 wall | **4,664.74 s（1:17:44.74）** | **6,660.65 s（1:51:00.65）** | **23,794.53 s（6:36:34.53）** |
| T50–T450 用时 | **2,695.25 s** | **2,914.44 s** | **19,645.77 s** |
| T50–T450 terminal throughput | **0.14841 task/s** | **0.13725 task/s** | **0.02036 task/s** |

按 T50–T450 主体段，Agentic-PD 的任务完成速率是原生单向 PD 的 **7.29×**，并比
Colocated 高 **8.13%**。Colocated 的 T500 被最后一题 `sympy__sympy-19040` 的
SWE-bench verifier 拉长：测试运行到统一的 2400 秒上限后记为 `command_timeout`。因此部署
性能主要比较 T50–T450；T500 仍保留为真实端到端评测 wall time。
原生组 T50–T450 中包含上述 1 个 infrastructure failure，因此这里严格称为 terminal
throughput；其对 400 条窗口的数值影响有限，但不能忽略其正确性差异。

## 3. T50–T450 主体窗口

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| 实际 Prefill compute | **4,660.78 token/s** | **4,993.35 token/s** | **14,052.34 token/s** |
| Decode throughput | **3,074.09 token/s** | **2,831.65 token/s** | **418.81 token/s** |
| 窗口内实际 Prefill | **12.56M tokens** | **14.55M tokens** | **276.07M tokens** |
| 窗口内 Decode | **8.29M tokens** | **8.25M tokens** | **8.23M tokens** |
| Forward / GPU | **P 78.34%；D 99.77%** | **Extend 15.88%；Decode 83.86%** | **P 99.98%；D 85.39%** |
| Prefill/服务排队 | **0.44** | **0.42 / 5组** | **104.63** |
| Decode running | **22.84/组；91.36/4组** | **21.23/组；106.17/5组** | **2.50/组；9.99/4组** |
| Decode transfer queue | **0/组** | **0/组** | **26.38/组；105.51/4组** |
| token-pool usage | **P 24.74%；D 79.32%** | **71.45% / 5组均值** | **P 4.74%；D 81.02%** |

原生单向 PD 的高 Prefill token/s 不是有效优势，而是 P 持续对完整历史做重复计算：P
Forward 接近 100%，平均排队 104.63；四个 D 组平均只有 9.99 个请求真正 running，另有
105.51 个已进入 Decode transfer queue。也就是说 1P 已成为供给瓶颈，D 大量时间在等
P→D handoff。Agentic-PD 通过 D→P 回传把主体段 Decode 提升到原生组的 **7.34×**，同时
消除了持续的 transfer queue。

Colocated 的 Extend/Decode Forward 是同一批 10 张 GPU 上的两类计算时间，不能解释为
独立 P/D 节点。修正后两组生成的工作量接近；Agentic-PD 的主体段 Decode token/s 比
Colocated 高 **8.56%**，任务完成速率高 **8.13%**。

## 4. 全程数据与 token 特性

“逻辑 Prompt”是每轮发送给模型的完整上下文累计；“实际 Prefill”来自 scheduler 计数器，
包含增量 Prefill 与重算。三次独立随机采样仍会产生不同轨迹，因此 token 工作量必须与
完成时间一起看；修正后的 Agentic-PD 与 Colocated 轨迹规模已经很接近。

| 指标 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| SWE-bench resolved | **254/500（50.8%）** | **247/500（49.4%）** | **246/500（49.2%）**；1题未评分 |
| 模型调用轮数 | **16,079** | **16,263** | **15,865** |
| 每题平均轮数 | **32.16** | **32.53** | **31.73** |
| 逻辑 Prompt 总量 | **335.02M** | **341.57M** | **329.60M** |
| GPU 实际 Prefill 总量 | **16.25M** | **16.88M** | **330.09M** |
| 实际 Prefill / 逻辑 Prompt | **4.85%** | **4.94%** | **100.15%** |
| Decode 总量 | **10.62M** | **10.61M** | **10.37M** |
| 每轮逻辑 Prompt | **20,836** | **21,003** | **20,775** |
| 每轮实际 Prefill | **1,011** | **1,038** | **20,806** |
| 每轮 Decode | **660** | **652** | **654** |
| 全程实际 Prefill compute | **3,483.34 token/s** | **2,533.58 token/s** | **13,872.71 token/s** |
| 全程 Decode throughput | **2,275.89 token/s** | **1,592.07 token/s** | **435.88 token/s** |

Agentic-PD 与原生组的轨迹规模接近：逻辑 Prompt 分别为 335.02M 和 329.60M、Decode
分别为 10.62M 和 10.37M。此时 Agentic-PD 只实际 Prefill 16.25M，而原生组实际 Prefill
330.09M，前者减少 **95.08%**，这直接量化了 D→P KV 回传的价值。原生组略高于 100%
来自 scheduler 计数边界、chunk 对齐和请求统计口径，不代表可复用 KV。

修正后的 Colocated 生成 16,263 轮和 10.61M Decode tokens，与 Agentic-PD 的 16,079 轮
和 10.62M Decode 基本一致。按每轮实际 Prefill 看，两种能复用 KV 的方法也接近
（1,011 对 1,038），远低于原生组的 20,806。Colocated 全程 token/s 被最后一个 2400 秒
verifier 长尾摊薄，因此计算阶段性能应结合 T50–T450 窗口查看。

旧 Colocated r1 使用 stock `pd_baseline` SGLang，出现 159 条结构化工具错误、244 个
`no_command`，最终只有 175/500 resolved，已作废。修正 r2 使用与 Agentic-PD 相同的
`pd` SGLang、关闭 Agentic KV 回传后，结构化工具错误降至 **13** 条，`no_command` 降至
**15**，平均轮数恢复为 **32.53**；这说明旧结果的低正确率和低轮数来自 baseline 的
多轮 Mamba/Radix 状态复用问题，而不是 Colocated 部署本身。

## 5. Agentic terminal/Host 验收

| 验收项 | Agentic-PD | Colocated | 原生单向 PD（无回传） |
|---|---:|---:|---:|
| `application_final` | **500×HTTP 200；503=0** | 不适用 | 不适用 |
| 最终 D→P Host | **0 entries / 0.0 GiB** | 不适用 | D→P 关闭 |
| 最终 `evicted_wait_child` | **0** | 不适用 | 不适用 |
| 最终 P→D pending/early/timer/phases | **全部为 0/空** | 不适用 | 原生 P→D 生命周期已结束 |
| 10秒采样 D→P Host 峰值 | **48 entries / 91.831 GiB** | 不适用 | 不适用 |
| D→P Host eviction commit | **318** | 不适用 | 不适用 |

Agentic 正式结果使用 r4。此前 r3 因 Router 缺少 group-control 环境变量导致 496 个
`application_final` 全部返回 503，Host/tombstone 泄漏，已作废。r4 中 500 个 terminal
edge 均到达固定 coordinator，最终双向 pending、Host 和 tombstone 全部清零。

## 6. 原始结果与统计口径

- Agentic-PD：`/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-1p4d-agentic-host512-64-swe500-c128-20261004-r4`
- Colocated：`/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-1plus4-colocated-pdcode-swe500-c128-20261004-r2`
- 原生单向 PD：`/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-1p4d-native-pd-swe500-c128-20261004-r2`
- 每组请求与轨迹为 `workload/requests.jsonl`，引擎指标为
  `workload/engine_metrics.jsonl` 和 `workload/engine_throughput_2s.jsonl`。
- T50–T450 使用各组第 50 和第 450 个 terminal timestamp 作为共同口径；counter 在边界
  做线性插值。Forward counter 按实际物理 GPU 数归一化。
- P/D token-pool usage 与 running gauge 先聚合实例，再按 TP2 组数给出均值；表中同时给出
  Decode running/transfer queue 的组均值和总和。
- 原生 r1 因 Apptainer/FUSE 超时清理挂起而作废；r2 使用修复后的有界 Host 清理完成全量。
  r2 结束后两节点均无残留 GPU compute process。
- Colocated stock-baseline r1 因 159 条结构化工具错误导致 244 个 `no_command`，不再作为
  正式结果；本表使用切换到 `pd` SGLang 且确认未启用 Agentic KV 回传的 r2。
