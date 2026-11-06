# Qwen3.5-122B-A10B TP8 双节点 SWE500：12 组统一结果

## 1. 范围与结论

本文件统一汇总三档并发（c128、c192、c256）和四种方法，共 12 组实验。
每组严格使用两个 H100 节点；PD 方法为一个 TP=8 Prefill 实例和一个 TP=8
Decode 实例，Colocated 为两个完整 TP=8 实例。数据集均为 SWE-bench Verified
500 题、source-order，主要指标为 T450。

**12/12 组均已得到可用 T450。**其中 11 组为 500/500、0 workload 失败；
c256 HiCache/Mooncake 返回了全部 500 个 terminal result，其中 491 题正常完成，
9 题因 SWE Apptainer 在进程峰值时 `fork` 返回 EAGAIN 而失败。P、D 和 Router
全程健康，因此保留其 T450，但不把它标成零异常的完整运行。

## 2. 统一核心结果

“T450 前 Decode 吞吐”定义为：从 workload 测量起点到第 450 个 terminal result
之间，所有 Decode endpoint 的单调 scheduler decode-token counter 增量除以该段墙钟
时间。它包含 T450 时仍在运行请求已经生成的 token，比只统计前 450 道已结束题目的
输出更准确地反映服务吞吐。多 endpoint 的 Colocated 结果先逐 endpoint 求 counter
增量再求和，避免临时 scrape 缺失造成聚合 counter 假下降。

| 并发 | 方法 | 正常完成 / 失败 | Resolved | T450 | T450 前 Decode tok/s | T450 前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Agentic-PD | 500 / 0 | 312 / 500 | 3,434.49 s | 1,566.64 | 5.380M |
| 128 | Colocated | 500 / 0 | 313 / 500 | **2,830.24 s** | **1,941.44** | 5.494M |
| 128 | 原生单向 PD（无 D→P 回传） | 500 / 0 | 316 / 500 | 11,462.79 s | 499.45 | 5.725M |
| 128 | 原生 HiCache/Mooncake PD | 500 / 0 | **322 / 500** | 3,616.09 s | 1,531.35 | 5.538M |
| 192 | Agentic-PD | 500 / 0 | 312 / 500 | **3,817.13 s** | **1,451.10** | 5.539M |
| 192 | Colocated | 500 / 0 | **317 / 500** | 4,711.84 s | 1,159.76 | 5.465M |
| 192 | 原生单向 PD（无 D→P 回传） | 500 / 0 | 298 / 500 | 11,188.71 s | 490.93 | 5.493M |
| 192 | 原生 HiCache/Mooncake PD | 500 / 0 | 310 / 500 | 9,720.99 s | 573.05 | 5.571M |
| 256 | Agentic-PD | 500 / 0 | 311 / 500 | **3,360.06 s** | **1,671.01** | 5.614M |
| 256 | Colocated | 500 / 0 | **317 / 500** | 6,801.94 s | 812.05 | 5.523M |
| 256 | 原生单向 PD（无 D→P 回传） | 500 / 0 | 313 / 500 | 11,399.05 s | 490.37 | 5.589M |
| 256 | 原生 HiCache/Mooncake PD | 491 / 9 | 301 / 500 | 11,099.96 s | 494.90 | 5.493M |

旧运行创建于 `measurement_start.json` 引入前时，表中 T450 保留对应正式报告的
既有值；新版运行直接使用测量起点标记。旧运行重新按首个稳定采样点核算 Decode
吞吐，起点差异仅为秒级，不影响表中两位小数的性能结论。

## 3. 全程计算量

下表的实际 Prefill 和 Decode 均来自 scheduler 单调 counter，而不是逻辑 Prompt
长度。它解释了为什么不同方法处理相近的 Decode 工作量，却有显著不同的 T450。

| 并发 | 方法 | 全程实际 Prefill | 全程 Decode | 全程平均 Prefill tok/s | 全程平均 Decode tok/s |
|---:|---|---:|---:|---:|---:|
| 128 | Agentic-PD | 17.38M | 5.55M | 3,741.16 | 1,194.55 |
| 128 | Colocated | 18.56M | 5.67M | 3,349.99 | 1,024.11 |
| 128 | 原生单向 PD | 529.97M | 5.88M | 39,416.43 | 436.97 |
| 128 | 原生 HiCache/Mooncake PD | 32.70M | 5.71M | 7,600.28 | 1,328.01 |
| 192 | Agentic-PD | 18.43M | 6.03M | 4,345.30 | 1,420.98 |
| 192 | Colocated | 222.62M | 5.67M | 29,636.50 | 755.26 |
| 192 | 原生单向 PD | 499.94M | 5.69M | 42,999.40 | 489.11 |
| 192 | 原生 HiCache/Mooncake PD | 277.02M | 5.81M | 27,260.27 | 572.08 |
| 256 | Agentic-PD | 17.51M | 5.67M | 3,564.07 | 1,153.48 |
| 256 | Colocated | 370.93M | 5.71M | 47,791.56 | 735.08 |
| 256 | 原生单向 PD | 508.76M | 5.67M | 43,170.28 | 480.73 |
| 256 | 原生 HiCache/Mooncake PD | 335.12M | 5.60M | 25,144.04 | 420.36 |

全程平均吞吐包含 T450 后有限 workload drain 和 verifier 长尾，因此服务能力比较优先
看上一表的 T450 及 T450 前 Decode 吞吐。

## 4. 直接观察

- c128 最快的是 Colocated：T450 为 2,830.24 秒，T450 前 Decode 为
  1,941.44 token/s；Agentic-PD 次之。
- c192 和 c256 最快的都是 Agentic-PD。其 T450 前 Decode 分别为
  1,451.10 和 1,671.01 token/s，均明显高于其他方法。
- 原生单向 PD 在三档并发下都执行约 500M Prefill token；它没有 D→P 回传，
  每轮几乎都重新计算完整历史，因此 Decode 长期只有约 490--499 token/s。
- Colocated 从 c128 的 18.56M Prefill 增长到 c192 的 222.62M、c256 的
  370.93M，说明并发提高后本地 cache 容量不足并触发大量重复 Prefill。
- HiCache/Mooncake 同样从 c128 的 32.70M 增至 c192 的 277.02M、c256 的
  335.12M；其高并发 T450 与 Decode throughput 明显退化。
- Agentic-PD 在三档并发下实际 Prefill 始终只有约 17--18M，说明扩大的
  D→P 512 GiB Host 与 P→D 256 GiB Host 能稳定保留跨工具轮次状态。

## 5. 可比性与异常说明

- 四种方法保持相同模型、数据顺序、采样参数、context 131,072、Prefill
  chunk/batch 8,192、`mem_fraction_static=0.8` 和双节点资源边界。
- Agentic-PD 使用 D→P 总 512 GiB、P→D 总 256 GiB Host，并启用 terminal
  cleanup；原生单向 PD 完全关闭 D→P 回传。
- HiCache c128/c192 使用 `mamba_full_memory_ratio=0.5`、Attention HiCache
  26 GiB/rank、Mamba HiCache 16 GiB/rank、Mooncake 512 GiB。c256 为通过原生
  baseline 的 Mamba admission 并维持 c256，使用 ratio 0.9、Attention 26 GiB/rank、
  Mamba HiCache 20 GiB/rank；没有修改 HiCache/Mooncake 的缓存或调度语义。
- c256 HiCache 的 9 个失败均为 verifier 容器进程压力导致的 EAGAIN；第 450 个
  terminal result 为 11,099.96 秒，第 450 个正常完成结果为 11,110.84 秒，二者仅差
  10.88 秒。

## 6. 原始结果目录

| 并发 | 方法 | 目录 |
|---:|---|---|
| 128 | Agentic-PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-agentic-1p1d-host512-256-swe500-c128-20261005-r1` |
| 128 | Colocated | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-colocated-2node-c128-swe500-20261002-r5` |
| 128 | 原生单向 PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-native-pd-tp8-swe500-c128-20261003-r1` |
| 128 | HiCache/Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-native-mooncake-tp8-swe500-c128-two-node-r3` |
| 192 | Agentic-PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-formal2n-agentic-1p1d-swe500-c192-20261011-r1` |
| 192 | Colocated | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-formal2n-colocated-2replica-swe500-c192-20261011-r1` |
| 192 | 原生单向 PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-formal2n-native-oneway-1p1d-swe500-c192-20261011-r1` |
| 192 | HiCache/Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-formal2n-hicache-mooncake-1p1d-swe500-c192-20261012-memopt-r1` |
| 256 | Agentic-PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-agentic-1p1d-host512-256-swe500-c256-20261005-r6` |
| 256 | Colocated | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-colocated-2node-c256-swe500-20261005-r1` |
| 256 | 原生单向 PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-native-pd-tp8-swe500-c256-20261006-r1` |
| 256 | HiCache/Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-122b-formal2n-hicache-mooncake-1p1d-swe500-c256-20261013-memopt-ratio09-mambahost20-dir-overlay-smallstack-r6` |
