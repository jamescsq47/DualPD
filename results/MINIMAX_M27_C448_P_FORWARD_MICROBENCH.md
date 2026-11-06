# MiniMax-M2.7 c448 P 侧固定 Shape 微基准

## 结论

Agentic-PD 在 T50–T450 内只执行了 HiCache/Mooncake **26.48%** 的新增
Prefill token（4.92M 对 18.56M），但 P Forward/GPU 时间反而为 HiCache 的
**107.23%**（721.49 s 对 672.83 s）。固定 shape 的本地恢复实验支持复用收益，
但**不能据此定量分解正式多节点 P Forward**。生产 V2 使用跨网 NIXL READ，
本实验使用本地 registered Host H2D，且未重放真实 token、逐请求长度和页碎片。
代码级诊断与修复方案见 [根因诊断报告](MINIMAX_M27_C448_P_FORWARD_ROOT_CAUSE.md)。

1. Agentic 的 batch 数只有 HiCache 的 **48.0%**，而不是 token 数对应的 26.5%；
   因为 Agentic 每批平均只有 2,662 个 new token，HiCache 为 4,822。MiniMax MoE
   每批有较高固定成本，所以 token 减少不会线性变成 Forward 时间减少。
2. 正式运行的平均每 batch GPU 时间为 Agentic **390.4 ms**、HiCache
   **174.8 ms**，Agentic 慢 **2.23×**。`0.480 × 2.233 = 1.072`，恰好解释
   为什么总 P Forward 是 107.2%。
3. 从稳态窗口各随机抽取 48 个真实聚合 shape 重放后，包含 cached prefix 和一条
   Host H2D 并行的预测为 Agentic **124.2 ms/batch**、HiCache
   **144.8 ms/batch**。换言之，固定 shape 数据面本来预测 Agentic 更快；正式
   Agentic 还有 **266.2 ms/batch** 残差，而 HiCache 只有 **29.9 ms/batch**。
4. 此前将 HiCache 的 29.9 ms/batch 当成共同生产 envelope，推导出
   236.3 ms/batch、436.7 s/GPU 的“Agentic 特有开销”。**撤回这一因果归属**：
   两种方法的残差未证明可交换，KNN 插值误差、真实 MoE routing 和传输实现差异
   均包含在其中。这些数字只是带额外假设的算术差，不能作为已测出的耗时构成。
5. 即便包含单路 DMA 干扰，恢复缓存仍远快于重算。Agentic p50 的完整上下文重算
   为 800.85 ms，而 KV resident+H2D 并行为 101.16 ms；问题不在“恢复是否值得”，
   而在生产运行中还有微基准未覆盖的并发传输/调度干扰。

因此，已确定的总量关系是 **batch 数减半，但正式每 batch 的 event 时间为 2.23 倍**。
这解释了算术关系，没有解释后者的全部因果来源。当前结果既不能证明“生产中重算
更划算”，也不能证明少 Prefill 的收益全部被 GPU 带宽竞争抵消。

## 正式运行对照

| 方法 | T50–T450 时长 | 新增 Prefill tok/s | 新增 token 总量 | P Forward/GPU | 每 GPU Forward 时间 | 8 GPU Forward 总量 |
|---|---:|---:|---:|---:|---:|---:|
| Agentic-PD | 1,133.01 s | 4,339.55 | 4.92M | 63.68% | 721.49 s | 5,771.88 GPU·s |
| HiCache/Mooncake | 1,042.55 s | 17,807.13 | 18.56M | 64.54% | 672.83 s | 5,382.66 GPU·s |

## 总量核账：为什么 Prefill 更少但 Forward 接近

| 项目 | Agentic-PD | HiCache | Agentic/HiCache |
|---|---:|---:|---:|
| T50–T450 P batch 数 | 1,848 | 3,850 | 0.480× |
| 平均 new token/batch | 2,662 | 4,822 | 0.552× |
| 平均 cached token/batch | 45,477 | 16,819 | 2.704× |
| 正式 GPU 时间/batch | **390.4 ms** | **174.8 ms** | **2.233×** |
| 真实 shape 微基准预测 | 124.2 ms | 144.8 ms | 0.858× |
| 正式值减微基准 | **266.2 ms** | **29.9 ms** | — |

124.2/144.8 ms 是窗口各 48 个聚合 shape 的合成重放再经 KNN 插值所得，
并非逐批实测预测真值。不能据此宣称修复后 P Forward 会降到 284.8 s/GPU。

### 插值残差与传输活动的相关性

以每个约 2 秒 metrics 区间中、每 rank `registered_host_extent` 完成数作为传输活动
代理，并用真实 shape KNN 微基准先扣除 batch/new/cached 差异：

| 每 rank 每区间 registered extent | batch 数 | 正式 ms/batch | shape 微基准预测 | 残差 |
|---:|---:|---:|---:|---:|
| 0 | 107 | 250.4 | 110.1 | 140.3 |
| 0–2 | 277 | 284.3 | 113.5 | 170.7 |
| 2–5 | 354 | 345.7 | 117.9 | 227.8 |
| 5–9 | 423 | 412.0 | 123.2 | 288.7 |
| 9–13 | 374 | 418.4 | 126.8 | 291.6 |
| 13–17 | 264 | 454.0 | 133.9 | 320.1 |
| >17 | 156 | 449.7 | 136.9 | 312.8 |

该表仅显示区间相关性；零 extent 区间不等于无 DMA，extent 注册完成数也不等于
传输占空比。此前用它分配“53% 传输开销”和“剩余约 204 s”没有因果依据，撤回
该分配。上表 0 行是 0–2 行的子集，不能再次加权相加。

未被识别的因素包括未被 extent 计数覆盖的
D→P Direct NIXL/P→D D2H、真实 per-request 长度相关性与 MoE expert imbalance，
以及 group controller/日志线程造成的 CPU kernel-launch gap。稳态窗口内 P 日志有
37,696 条 registered extent、4,817 次 group-attempt abort 和 643 条慢尾 handoff；
这些控制/数据面活动是单进程微基准没有重放的。要把剩余部分严格归因，需要下次
短测同时记录逐 batch per-request lens、expert-per-rank token 数、各 DMA active
interval 和 Forward kernel gap。

## 真实 shape 分布与采样点

shape 是 P service log 的聚合 `(batch size, new-token total, cached-prefix total)`。
Agentic 共 2,975 个 Forward batch，HiCache 共 6,019 个。

| 方法 | 指标 | 均值 | p25 | p50 | p75 | 选取的真实联合 shape |
|---|---|---:|---:|---:|---:|---|
| Agentic | batch size | 3.79 | 1 | 3 | 5 | p25 `(1,704,14656)`；p50 `(3,1536,29504)`；p75 `(5,3008,49088)` |
| Agentic | new token | 2,247 | 704 | 1,536 | 3,136 | 同上 |
| Agentic | cached prefix | 34,014 | 14,336 | 29,504 | 50,368 | 同上 |
| HiCache | batch size | 2.00 | 1 | 1 | 2 | p25 `(1,576,640)`；p50 `(1,2304,7744)`；p75 `(2,7616,17280)` |
| HiCache | new token | 3,577 | 640 | 2,176 | 7,488 | 同上 |
| HiCache | cached prefix | 13,386 | 576 | 7,424 | 18,560 | 同上 |
| HiCache | 高频重算块 | — | — | — | — | mode `(1,8192,256)`，出现 784 次 |

选取的是日志中距离各边际分位点最近的**真实聚合三元组**，不是分别拼接三个分位
数。SGLang 日志不含 batch 内逐请求长度向量，因此微基准在保持聚合三元组完全一致
的前提下，把长度按 page=64 均衡分给请求；这部分是合成的，不能用于研究 batch 内
长度方差。

## 固定 shape 结果

下表均为 9 次重复的 TP8 rank-max 中位数，单位为 ms。`new-only` 是使用相同 suffix
token、但没有 prefix 的反事实基线；MiniMax 是 MoE，prefix 会改变 hidden state、
expert routing 和 collective，因此 `resident - new-only` 只能作为 cached-KV/长前缀
Attention 路径的**混合残差**，不是纯 Attention kernel 时间，也不是其上界。
下表对负差截断为零仅沿用原输出格式，不能消除 MoE 混杂。DMA 竞争采用
`resident/overlap` ABBA 交错的逐对差值中位数；各列的独立中位数不要求严格相加。

| Shape | B/new/cached | 完整上下文重算 | new-only 计算 | KV 已驻留 Forward | 长前缀路径混合残差 | KV DMA∥Forward | 成对本地 H2D 干扰 | 重叠结束后的外部等待 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Agentic p25 | 1/704/14,656 | 364.09 | 73.27 | 73.81 | 0.54 | 83.55 | **9.23** | 0.69 |
| Agentic p50 | 3/1,536/29,504 | 800.85 | 82.49 | 85.60 | 3.11 | 101.16 | **13.99** | 0.72 |
| Agentic p75 | 5/3,008/49,088 | 1,326.65 | 84.84 | 92.83 | 7.99 | 122.58 | **22.29** | 0.64 |
| HiCache p25 | 1/576/640 | 76.03 | 75.74 | 75.38 | 0.00 | 75.79 | **0.31** | 0.58 |
| HiCache p50 | 1/2,304/7,744 | 269.50 | 86.64 | 86.39 | 0.00 | 91.51 | **5.11** | 0.68 |
| HiCache p75 | 2/7,616/17,280 | 596.20 | 167.24 | 187.76 | 20.52 | 191.37 | **3.54** | 0.65 |
| HiCache mode | 1/8,192/256 | 255.61 | 182.26 | 182.95 | 0.69 | 183.86 | **0.93** | 0.64 |

“重叠结束后的外部等待”是微基准 overlap makespan 减 TP-max Forward wall time，表示
Forward 结束后尚未完成的 DMA tail 和线程启动开销，不是完整的
admission→scheduler-ready→Forward 等待。

## Host H2D 恢复

微基准使用 legacy 本地路径的 1,024-token 逐 chunk pipeline、独立 KV 区域、registered Host
memory、H2D stream priority=-1，并与 Forward stream 并行。生产请求值为 priority
-5，但 H100 实际支持范围把它钳制为 -3；输出记录的是有效 priority。

| Shape | 恢复 token | 单 rank 数据量 | Host→HBM 时间 | 单 rank 带宽 | 8 rank 算术聚合带宽 |
|---|---:|---:|---:|---:|---:|
| Agentic p25 | 14,656 | 0.433 GiB | 21.63 ms | 20.03 GiB/s | 160.2 GiB/s |
| Agentic p50 | 29,504 | 0.872 GiB | 42.77 ms | 20.39 GiB/s | 163.1 GiB/s |
| Agentic p75 | 49,088 | 1.451 GiB | 68.44 ms | 21.20 GiB/s | 169.6 GiB/s |
| HiCache p25 | 640 | 0.019 GiB | 1.15 ms | 16.43 GiB/s | 131.4 GiB/s |
| HiCache p50 | 7,744 | 0.229 GiB | 11.72 ms | 19.53 GiB/s | 156.2 GiB/s |
| HiCache p75 | 17,280 | 0.511 GiB | 24.97 ms | 20.46 GiB/s | 163.7 GiB/s |
| HiCache mode | 256 | 0.008 GiB | 0.61 ms | 12.34 GiB/s | 98.7 GiB/s |

这里量化的是本地 registered Host→P HBM；**不是生产 V2 的跨节点 NIXL READ**，
也不是原生 HiCache 恢复实现。HiCache 行只采用其 shape，不能将本表作为两种方法
生产传输带宽的对照，或将 contention 数值直接泛化为 NIXL contention。

## 生产 Host/NIXL 慢尾恢复

生产 trace 只在 ownership handoff 总耗时不少于 1 秒时打印，所以下表是
**慢尾条件统计**，不是全体恢复分布。transfer phase 是 coordinator 从
PREPARED 到所有 rank DMA_DONE 的时间，包含命令广播/调度，带宽是 effective
phase bandwidth，不是纯 NIXL/PCIe device 带宽。non-DMA 只含
PREPARE+BIND+RELEASE，是完整 Forward 外等待的下界。

| D→P 路径 | 慢尾样本 | total p50 / p90 | transfer phase p50 / p90 | effective 带宽 p50 / p90 | ownership non-DMA p50 / p90 |
|---|---:|---:|---:|---:|---:|
| Direct/NIXL | 118 | 1.203 / 1.695 s | 1.026 / 1.358 s | 4.16 / 6.49 GiB/s | 0.104 / 0.648 s |
| Host restore | 251 | 1.169 / 1.560 s | 0.978 / 1.247 s | 4.96 / 12.73 GiB/s | 0.130 / 1.005 s |

当前日志没有可按 generation key 关联的 scheduler-ready→Forward-start 事件，因此
不能诚实地给出完整 Forward 外排队时间；上表 non-DMA 是可观测下界，微基准的
0.58–0.72 ms 则只是 DMA 尾部等待。要补全这一项，需要在生产 scheduler 中增加
不改变调度语义的 Forward-start trace。

## 配置、原始数据与复现

- 模型：MiniMax-M2.7 revision `d494266a`，官方 block-FP8 权重，BF16 activation/KV。
- SGLang：`4b353f9142db666dd95664fddc1346b012022197`。
- GPU：1×H100 节点，TP=8、EP=8；FA3；page=64；context=131072；Prefill chunk=8192。
- 微基准原始结果：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-c448-p-microbench/full-production-r4.json`。
- 生产恢复 trace 汇总：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-c448-p-microbench/production-restore-trace.json`。
- 微基准脚本：
  `slime/examples/pd/scripts/bandwidth/benchmark_minimax_p_forward_decomposition.py`。
- trace 汇总脚本：
  `slime/examples/pd/scripts/bandwidth/summarize_minimax_agentic_restore_trace.py`。

两份脚本经独立审计后才进入正式 TP8 采样；它们不接入 controller/ledger，不改变
ownership 或 runtime 状态机。
