# 九组模型实验的 Prefill Token 统计与减少量

本文统一整理九组 Agentic-PD 实验中的实际 Prefill 计算量，并比较当前新方法相对
Colocated baseline 与 No-reverse PD 减少的 Prefill token。

## 统计口径

- `Baseline` 指各原始结果文档中明确命名的 **Colocated baseline**。
- 同时列出 `No-reverse`：它代表完全没有 D→P 回传、每轮需要重新 Prefill 历史的
  原生单向 PD。
- BrowseComp 为固定 300 秒预热 + 1,200 秒正式稳态窗口。各方法完成的 Agent 数不同，
  因而使用 **实际 Prefill tokens / 完成 Agent** 进行等工作量比较。
- SWE-bench Verified 只报告 **实际跑完500个terminal的全程 Prefill token**：直接用
  workload 全程 `scheduler_prefill_compute_tokens_per_second × measurement_elapsed_seconds`，
  并以 `completed + failed = 500` 为硬门槛。T450主动停止、缺少完整summary或缺少完整
  scheduler counter的运行统一标为“缺失”，不做400→500线性换算。
- “理论需要 Prefill”按用户指定的三组平均口径估计：取该模型三档并发下新方法的
  **实际500题全程 Prefill token平均值**；只有三档都有完整500题实测时才填写。
  Qwen3.5-35B-A3B使用主矩阵c64/c128/c192，额外c256不参与均值。该列是三次实际
  观测的经验基准，不是逐token LCP证明出的严格理论下界。
- “少算”是两个方法实际物理 Prefill 计算量之差。它包含缓存驱逐、重复历史计算以及
  少量轨迹差异，不冒充严格理论下界意义上的“额外 token”。
- BrowseComp 的 Qwen3-32B 与 Qwen3-30B-A3B Prefill/Agent 由正式窗口
  `Prefill tok/s ÷ Agent/s` 得到，因此受原表显示精度影响，为近似值。

用户口述名称与本文实验组的映射：

- `qwen3,8-27b`：Qwen3.8-27B 单节点 2P:2D；
- `qwen3.5-25b`：Qwen3.5-35B-A3B；
- 后一个 `qwen3.8-27b`：Qwen3.8-27B 双节点 2P:6D。

## 1. Qwen3-8B · BrowseComp

原始结果：[`BROWSECOMP_QWEN3_8B_H100.md`](BROWSECOMP_QWEN3_8B_H100.md)。
单位为 K Prefill tokens/Agent。

| 并发 | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 14.18K | 14.10K | 45.43K | 28.44K | -0.08K（-0.6%） | 31.25K（68.8%） |
| 128 | 13.55K | 14.20K | 46.20K | 32.06K | 0.65K（4.6%） | 32.64K（70.7%） |
| 256 | 13.41K | 18.19K | 45.96K | 36.45K | 4.79K（26.3%） | 32.55K（70.8%） |
| 384 | 13.95K | 19.96K | 46.55K | 39.38K | 6.01K（30.1%） | 32.60K（70.0%） |
| 512 | 15.05K | 25.65K | 45.39K | 40.38K | 10.61K（41.4%） | 30.34K（66.9%） |
| 576 | 15.01K | 29.34K | 43.14K | 40.54K | 14.32K（48.8%） | 28.13K（65.2%） |

随着并发提高，Colocated 的缓存驱逐变得明显。新方法相对 Colocated 的 Prefill
节省由近似 0 提高到 48.8%；相对 No-reverse 始终减少约 65%–71%。

## 2. Qwen3-32B TP=2 · BrowseComp

原始结果：[`BROWSECOMP_QWEN3_32B_TP2_H100.md`](BROWSECOMP_QWEN3_32B_TP2_H100.md)。
单位为 K Prefill tokens/Agent。

| 并发 | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 8.93K | 17.35K | 18.66K | 17.40K | 8.42K（48.6%） | 9.73K（52.2%） |
| 256 | 9.34K | 17.01K | 17.53K | 16.65K | 7.68K（45.1%） | 8.20K（46.7%） |
| 320 | 9.92K | 16.90K | 17.54K | 18.25K | 6.98K（41.3%） | 7.62K（43.4%） |

新方法稳定只执行约 9–10K Prefill/Agent；相比 Colocated 少约 41%–49%，相比
No-reverse 少约 43%–52%。

## 3. Qwen3-30B-A3B TP=2 · BrowseComp

原始结果：[`BROWSECOMP_QWEN3_30B_A3B_TP2_H100.md`](BROWSECOMP_QWEN3_30B_A3B_TP2_H100.md)。
单位为 K Prefill tokens/Agent。

| 并发 | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6.86K | 6.84K | 15.09K | 6.29K | -0.02K（-0.3%） | 8.23K（54.5%） |
| 192 | 7.01K | 7.30K | 15.82K | 6.40K | 0.29K（4.0%） | 8.81K（55.7%） |
| 256 | 6.99K | 6.74K | 16.00K | 7.94K | -0.24K（-3.6%） | 9.02K（56.3%） |
| 320 | 6.58K | 7.59K | 14.91K | 9.81K | 1.01K（13.3%） | 8.34K（55.9%） |
| 384 | 6.63K | 7.89K | 16.33K | 10.51K | 1.26K（16.0%） | 9.70K（59.4%） |

低并发时 Colocated 尚未出现明显缓存压力，新方法与其基本持平；到 c320/c384 才减少
13%–16%。相对 No-reverse 则一直减少约 55%–59%。

## 4. Qwen3.5-9B TP=1 · SWE-bench Verified

原始结果：[`SWEBENCH_QWEN35_9B_TP1_H100_MILES_PR51.md`](SWEBENCH_QWEN35_9B_TP1_H100_MILES_PR51.md)。
单位为 M Prefill tokens/实际500题全程。

| 并发 | 理论需要（三档均值） | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 11.148M | 11.006M | 18.337M | 281.596M | 70.539M | 7.332M（40.0%） | 270.591M（96.1%） |
| 192 | 11.148M | 11.194M | 26.943M | 287.811M | 124.022M | 15.749M（58.5%） | 276.617M（96.1%） |
| 256 | 11.148M | 11.243M | 33.260M | 274.198M | 172.800M | 22.017M（66.2%） | 262.954M（95.9%） |

新方法的 Prefill 工作量基本不随并发增加；Colocated 和 HiCache 则因缓存压力快速
上升。相对 No-reverse，新方法消除了约 96% 的重复 Prefill。

## 5. Qwen3.8-27B TP=2 · 单节点 2P:2D

原始结果：[`SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_2P2D_H100.md`](SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_2P2D_H100.md)。
单位为 M Prefill tokens/实际500题全程。

本轮九组没有重新运行 Colocated，因此直接比较新方法、No-reverse 与 HiCache。
c128/c192 HiCache 使用原文中的更新版官方 SGLang 补测。

| 并发 | 理论需要（三档均值） | 新方法 | No-reverse | HiCache | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|
| 96 | 缺失 | 17.040M | 370.681M | 缺失 | 353.642M（95.4%） |
| 128 | 缺失 | 缺失 | 缺失 | 145.016M | 缺失 |
| 192 | 缺失 | 缺失 | 缺失 | 162.428M | 缺失 |

c96新方法与No-reverse有完整500题实测，新方法少算95.4%。c96 HiCache以及
c128/c192新方法与No-reverse均在T450停止；虽然原文有T450窗口数据，但这里按
500题硬口径标为缺失。三档新方法不足，因此理论三组均值也标为缺失。

## 6. Qwen3.5-35B-A3B TP=2 · 单节点 2P:2D

原始结果：[`SWEBENCH_QWEN35_35B_A3B_TP2_SINGLE_NODE_2P2D_H100.md`](SWEBENCH_QWEN35_35B_A3B_TP2_SINGLE_NODE_2P2D_H100.md)。
单位为 M Prefill tokens/实际500题全程。

| 并发 | 理论需要（三档均值） | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 17.572M | 17.632M | 19.897M | 510.865M | 14.486M | 2.264M（11.4%） | 493.233M（96.5%） |
| 128 | 17.572M | 17.790M | 57.617M | 503.756M | 29.517M | 39.827M（69.1%） | 485.965M（96.5%） |
| 192 | 17.572M | 17.293M | 85.514M | 522.005M | 70.321M | 68.222M（79.8%） | 504.712M（96.7%） |
| 256 | 17.572M | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |

该组的并发效应非常明显：c64/c128/c192新方法始终约17–18M，而Colocated从
19.90M增至85.51M；c192时新方法比Colocated少算79.8%。c256四种方法均只保留到
T450，因此不填500题全程值。

## 7. Qwen3.8-27B TP=2 · 双节点 2P:6D

原始结果：[`SWEBENCH_QWEN38_27B_TP2_2P6D_H100_MATRIX.md`](SWEBENCH_QWEN38_27B_TP2_2P6D_H100_MATRIX.md)。
单位为 M Prefill tokens/实际500题全程。

| 并发 | 理论需要（三档均值） | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16.283M | 16.234M | 15.895M | 343.733M | 195.616M | -0.339M（-2.1%） | 327.499M（95.3%） |
| 192 | 16.283M | 16.419M | 17.407M | 缺失 | 缺失 | 0.988M（5.7%） | 缺失 |
| 256 | 16.283M | 16.195M | 34.501M | 缺失 | 163.628M | 18.307M（53.1%） | 缺失 |

c192/c256的旧版No-reverse与c192 HiCache只保留了T450结果，不能可靠补出500题
全程token总量。该组相对Colocated从c128略多算2.1%，到c256少算53.1%；c128可验证
相对No-reverse减少95.3%。

## 8. Qwen3.5-122B-A10B TP=8 · 双节点

原始结果：[`SWEBENCH_QWEN35_122B_A10B_TP8_C128_C192_C256_MATRIX.md`](SWEBENCH_QWEN35_122B_A10B_TP8_C128_C192_C256_MATRIX.md)。
单位为 M Prefill tokens/实际500题全程。

| 并发 | 理论需要（三档均值） | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 17.775M | 17.385M | 18.556M | 529.970M | 32.696M | 1.171M（6.3%） | 512.585M（96.7%） |
| 192 | 17.775M | 18.428M | 222.622M | 499.943M | 277.023M | 204.194M（91.7%） | 481.515M（96.3%） |
| 256 | 17.775M | 17.511M | 370.926M | 508.761M | 335.121M | 353.415M（95.3%） | 491.251M（96.6%） |

c128 时 Colocated 缓存仍够用；到 c192/c256 后出现严重缓存压力。c256 时新方法只
执行17.51M，而Colocated执行370.93M、No-reverse执行508.76M。

## 9. MiniMax-M2.7 TP=8/EP=8 · 双节点

原始结果：[`SWEBENCH_MINIMAX_M27_TP8_TWO_NODE.md`](SWEBENCH_MINIMAX_M27_TP8_TWO_NODE.md)。
单位为 M Prefill tokens/实际500题全程。

| 并发 | 理论需要（三档均值） | 新方法 | Colocated | No-reverse | HiCache | 相比 Colocated 少算 | 相比 No-reverse 少算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |
| 192 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |
| 256 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |

这组正式矩阵以T450为停止条件，明确不要求T500；三档都没有可用于本表的完整500题
全程counter。因此全部标为缺失，不再把T50–T450窗口线性外推到500题。

## 总体结论

- 相对 No-reverse，BrowseComp 三组减少约 **43%–71%** 的实际 Prefill；在具有
  完整500题双方实测的 SWE-bench Qwen 单元中减少约 **95.3%–96.7%**。
- 相对 Colocated，收益主要取决于缓存压力。低并发下可能基本持平；高并发下
  Qwen3.5-35B-A3B（c192）和 Qwen3.5-122B-A10B（c256）分别减少 **79.8%** 和
  **95.3%**。
- 在当前具有完整500题实测的新方法运行中，实际 Prefill 约为 **11–18.4M tokens**；
  No-reverse则约为 **274–530M tokens**。MiniMax-M2.7没有500题全程实测，不进入
  该范围。
- Qwen3-30B-A3B 的低并发 Colocated 尚未受到明显缓存压力，因此新方法在 c128/c256
  的 Prefill 量与 Colocated 基本持平，并没有虚构为正向节省。
