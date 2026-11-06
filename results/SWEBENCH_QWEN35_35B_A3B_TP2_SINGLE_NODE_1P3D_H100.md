# SWE-bench Verified · Qwen3.5-35B-A3B TP=2 · 单节点H100 · 1P:3D矩阵

> 创建：2026-10-14。仅使用`node2800`串行完成12组实验；每组通过零残留门禁后
> 自动启动下一组。本矩阵只把27B矩阵的模型替换为Qwen3.5-35B-A3B。

## 实验范围

- 模型：`Qwen/Qwen3.5-35B-A3B`，HF revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`，TP=2，BF16；实验context仍固定
  为131,072，不使用模型更长的原生窗口改变负载。
- 数据：SWE-bench Verified固定500题，source-order，seed 2026。
- Harness：`openenv_structured_tool`；最多64轮、每轮最多8,192 token，
  temperature 0.6、top-p 0.95、top-k 20。
- 并发：64、128、192；每档依次运行当前新方法、Colocated、No-reverse PD、
  原生HiCache/Mooncake，共12组。
- 单节点布局：`P0=GPU0-1`、`D0=GPU2-3`、`D1=GPU4-5`、
  `D2=GPU6-7`；Colocated使用4个TP2完整副本。
- 当前新方法：D→P Host总量512GiB（D0每rank 86GiB，D1/D2每rank 85GiB），
  P→D Host总量256GiB（P0每rank 128GiB），每个D的共享Decode growth reserve
  为131,072 tokens。
- Colocated与No-reverse使用隔离的`b1ef6` baseline，仅含既有GDN contiguous、
  原生Mamba slot admission正确性修复，以及Qwen3.5在Transformers v5下所需的
  config子类初始化兼容；不含新方法代码。HiCache沿用baseline SGLang、Mooncake
  原生机制和256GB共享segment，并使用同一项纯模型加载兼容。

## 核心指标口径

- **T450**：从workload测量起点到第450个terminal result的墙钟时间。
- **T450前Decode吞吐**：同一时间段内各Decode endpoint的
  `sglang_realtime_tokens_total{mode="decode"}`增量之和除以T450；各endpoint
  独立做差后求和，scrape边界使用线性插值。
- 全程吞吐包含T450后的有限任务drain和verifier长尾，仅作为辅助指标。

## 串行与清理约束

执行顺序为`c64 → c128 → c192`，每档按“当前新方法 → Colocated →
No-reverse → HiCache”。每组切换前必须确认GPU进程、矩阵端口、Apptainer、
`squashfuse_ll`和`faked-sysv`全部归零；清理失败时只能取消并重新申请
`node2800`，不得改用其他节点或启动下一组。

<!-- AUTOPILOT_RESULTS_START -->
## 自动运行进度与结果

> 最后检查：2026-10-16 07:57:41 UTC。
> 状态：`paused-after-T450`；有效T450结果 **11/12**。
> c192 No-reverse在第493个terminal后按要求停止（正常完成489、失败4、resolved 256、environment error 0）；T450窗口完整保留。c192 HiCache尚未启动。

### 核心指标

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法（1P:3D） | 500/0 | 279/500 | **3,399.43** | **1,508.98** | 5.130M |
| 64 | Colocated（4×TP2） | 500/0 | 273/500 | **2,668.89** | **2,050.97** | 5.474M |
| 64 | No-reverse PD（1P:3D） | 500/0 | 274/500 | **16,323.71** | **324.77** | 5.301M |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 500/0 | 287/500 | **3,311.97** | **1,627.29** | 5.390M |
| 128 | 当前新方法（1P:3D） | 499/1 | 271/500 | **3,534.76** | **1,472.80** | 5.206M |
| 128 | Colocated（4×TP2） | 500/0 | 281/500 | **2,206.11** | **2,476.64** | 5.464M |
| 128 | No-reverse PD（1P:3D） | 499/1 | 270/500 | **16,942.45** | **309.88** | 5.250M |
| 128 | 原生 HiCache/Mooncake（1P:3D） | 500/0 | 289/500 | **2,847.86** | **1,861.40** | 5.301M |
| 192 | 当前新方法（1P:3D） | 500/0 | 276/500 | **3,840.60** | **1,410.18** | 5.416M |
| 192 | Colocated（4×TP2） | 500/0 | 282/500 | **2,217.44** | **2,425.71** | 5.379M |
| 192 | No-reverse PD（1P:3D） | 489/4（T450后停止） | 256/493 | **16,339.62** | **317.21** | 5.183M |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — |

### 全程与辅助指标

| 并发 | 方法 | T50 | T250 | T500 | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 当前新方法（1P:3D） | 490.69 | 1,925.09 | 5,901.81 | 0.13752 | 2,943.92 | 921.41 | 26 | — |
| 64 | Colocated（4×TP2） | 449.64 | 1,530.60 | 3,435.92 | 0.18024 | 5,790.10 | 1,672.68 | 12 | — |
| 64 | No-reverse PD（1P:3D） | 2,471.21 | 9,409.29 | 17,393.45 | 0.02888 | 28,594.71 | 321.97 | 62 | — |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 448.65 | 1,757.80 | 5,642.31 | 0.13970 | 2,674.17 | 989.01 | 41 | — |
| 128 | 当前新方法（1P:3D） | 765.69 | 2,233.27 | 5,774.92 | 0.14445 | 2,980.31 | 931.87 | 20 | env=1, failed=1 |
| 128 | Colocated（4×TP2） | 433.05 | 1,273.60 | 4,946.82 | 0.22560 | 11,647.53 | 1,132.48 | 30 | — |
| 128 | No-reverse PD（1P:3D） | 4,422.96 | 10,450.08 | 17,690.28 | 0.03195 | 28,599.75 | 306.63 | 127 | env=1, failed=1 |
| 128 | 原生 HiCache/Mooncake（1P:3D） | 612.13 | 1,731.27 | 3,485.23 | 0.17891 | 5,562.63 | 1,584.80 | 111 | — |
| 192 | 当前新方法（1P:3D） | 1,173.08 | 2,447.99 | 6,069.40 | 0.14995 | 2,902.77 | 915.98 | 47 | — |
| 192 | Colocated（4×TP2） | 519.91 | 1,454.17 | 4,686.72 | 0.23564 | 18,243.77 | 1,160.95 | 36 | — |
| 192 | No-reverse PD（1P:3D） | 4,002.25 | 10,003.57 | — | 0.03242 | — | — | — | T450有效；493/500后按要求停止，failed=4 |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — | — | — | 待完成 |

### T450窗口 P/D Forward、KV与复用

| 并发 | 方法 | P Forward/卡 | P Attention KV/引擎 | D Forward/卡 | D/共享 Attention KV/引擎 | Prefix复用率 |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法（1P:3D） | 70.19% | 7.48% | 98.77% | 8.47% | 96.40% |
| 64 | Colocated（4×TP2） | 28.17% | — | 70.57% | 11.68% | 96.09% |
| 64 | No-reverse PD（1P:3D） | 99.87% | 1.93% | 62.07% | 24.46% | 0.00% |
| 64 | 原生 HiCache/Mooncake（1P:3D） | 59.23% | 5.46% | 93.18% | 18.69% | 96.99% |
| 128 | 当前新方法（1P:3D） | 68.99% | 7.29% | 98.89% | 8.42% | 96.41% |
| 128 | Colocated（4×TP2） | 44.09% | — | 55.10% | 22.82% | 88.45% |
| 128 | No-reverse PD（1P:3D） | 99.90% | 1.95% | 59.89% | 49.98% | 0.00% |
| 128 | 原生 HiCache/Mooncake（1P:3D） | 61.99% | 10.61% | 94.61% | 40.89% | 96.08% |
| 192 | 当前新方法（1P:3D） | 66.72% | 7.63% | 98.20% | 8.37% | 96.48% |
| 192 | Colocated（4×TP2） | 53.93% | — | 44.73% | 32.39% | 82.80% |
| 192 | No-reverse PD（1P:3D） | 99.84% | 1.95% | 60.73% | 72.77% | 0.00% |
| 192 | 原生 HiCache/Mooncake（1P:3D） | — | — | — | — | — |

Forward为从workload起点到T450的物理GPU平均device-timer占比；
Attention KV为同窗口内`sglang_full_token_usage`的时间加权均值，按逻辑TP=2引擎平均。
Colocated共享P/D KV pool，因此P KV留空，共享值记在D/共享列。
Prefix复用率为P侧GPU token计数器`prefill_cache / (prefill_cache + prefill_compute)`；这是实际prefix命中口径，不冒充仅统计page-aligned parent-KV的专用回传成功率。

原始目录：

- c64 当前新方法（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-agentic-swe500-c64-20261014-r1`
- c64 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-colocated4-swe500-c64-20261014-correctness-r1`
- c64 No-reverse PD（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-native-oneway-swe500-c64-20261014-correctness-r1`
- c64 原生 HiCache/Mooncake（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-hicache-mooncake-swe500-c64-20261014-r2`
- c128 当前新方法（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-agentic-swe500-c128-20261014-r1`
- c128 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-colocated4-swe500-c128-20261014-correctness-r1`
- c128 No-reverse PD（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-native-oneway-swe500-c128-20261014-correctness-r1`
- c128 原生 HiCache/Mooncake（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-hicache-mooncake-swe500-c128-20261014-r2`
- c192 当前新方法（1P:3D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-agentic-swe500-c192-20261014-r1`
- c192 Colocated（4×TP2）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-colocated4-swe500-c192-20261014-correctness-r1`
- c192 No-reverse PD（1P:3D，T450后停止）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-h100-1n-1p3d-native-oneway-swe500-c192-20261014-correctness-r1`
<!-- AUTOPILOT_RESULTS_END -->

## D→P恢复优化

为隔离D→P Host恢复/准入瓶颈，先在c128复用完全相同的TP2、1P:3D、
512GB D→P Host、256GB P→D Host和131,072-token Decode growth配置，
只修改`host_lanes: 2→4`、`host_load_workers: 1→2`、
`fast_tool_seconds: 1→2`。试验在第450个terminal后自动停止，以原生HiCache
c128的1,861.40 Decode tok/s为最低通过线。

- v1结果：T450 **3,423.34s**，T450 Decode **1,575.79 tok/s**，为HiCache的
  **84.66%**；相对原新方法的1,472.80 tok/s提升**7.0%**，但未达到通过线。
- v1资源：P/D Forward 68.53%/98.34%，P/D KV 11.23%/11.45%，
  Prefix复用率96.46%。
- v1结论：`admission_pending`均值从76.9降至59.2，但双H2D worker造成
  大量Host restore DMA落在约1.1–1.5秒，继续增加物理恢复并行度不是正确方向。
- v2计划：恢复`host_lanes=2`、`host_load_workers=1`，仅将
  `fast_tool_seconds`提高到5秒，利用低D KV占用消除绝大多数Host往返。
- run：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-hl4hw2ft2-r1`
- 控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-tuning-20261016/state.json`
- v2结果：T450 **3,118.38s**，T450 Decode **1,711.30 tok/s**，为HiCache的
  **91.94%**；相对原新方法提升**16.2%**，仍未达到通过线。
- v2资源：P/D Forward 68.44%/97.21%，P/D KV 15.04%/18.45%，
  D KV峰值39.12%，Prefix复用率96.48%。P/D请求率8.42/8.48 req/s，
  每个D的平均running batch为7.22；HiCache对应9.31/9.41 req/s和8.58。
- v3计划：仅将`fast_tool_seconds`从5秒提高至10秒。若仍未追平，停止阈值调参，
  转入Host durable后在工具执行期间预恢复到P。
- v2 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft5-r2`
- v2控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft5-20261016/state.json`
- v3结果：T450 **3,168.28s**，T450 Decode **1,720.93 tok/s**，为HiCache的
  **92.45%**；相对v2仅再提升**0.56%**，相对原新方法提升**16.85%**，
  仍低于通过线**7.55%**。正常完成461、失败0、resolved 257，在T450后清理完成。
- v3资源：P/D Forward 73.92%/97.85%，P/D KV 11.52%/28.94%，
  Prefix复用率96.44%。提高Direct保留窗口明显增加了P工作和D驻留KV，
  但吞吐收益已饱和，因此不再继续提高`fast_tool_seconds`。
- v3 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft10-r3`
- v3控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft10-20261016/state.json`
- v4验证：保持`host_lanes=2`和`host_load_workers=1`，将Direct保留窗口
  提高至30秒，验证用D端低使用率KV直接覆盖长工具间隔是否能消除剩余Host H2D气泡。
  这仍是显式的单次实验配置，不改数据面默认行为；若D KV接近已证实安全的
  HiCache水位仍不达标，则转入“复用P上已有Radix前缀、仅恢复D新增KV和Mamba
  checkpoint”的增量恢复实现。
- v4部分结果：到T250时Decode **2,118.56 tok/s**，比HiCache的
  1,861.40 tok/s高**13.82%**，说明30秒Direct保留已能消除主要D→P供给气泡。
  运行在284 terminal时因`MemAvailable=232,573,536 KiB`低于原256GiB安全门禁
  而停止，清理期间最终保留305条记录；该run未达T450，不作为最终成绩。
- v4 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft30-r4`
- v4控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft30-20261016/state.json`
- v5恢复方案：保持c128、30秒Direct窗口、512/256GiB Host容量和全部模型/
  workload配置不变。启动硬门禁通过后，不再每分钟fork新的ssh/srun
  资源探针，改以共享进度文件作为存活信号；20分钟无terminal增长由watchdog
  报警。T450结果落盘后直接取消allocation，由Slurm cgroup统一回收，
  然后重建并通过零残留门禁。v4实际峰值由768GiB固定Host注册区、约1TiB
  workload step及模型进程共同构成，已接近原`--mem=1800G` cgroup上限；v5
  因而在同一node2800改用`--mem=0`整机内存allocation（job `26061423`，
  Slurm分配`2,000,000M`），不降低Host容量或并发。2026-10-16 12:41 UTC
  四个TP2实例均已就绪并重新启动workload。
- v5 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft30-r5`
- v5控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft30-r5-20261016/state.json`
- v5结果：T450 **3,149.81s**，T450 Decode **1,687.98 tok/s**，为HiCache的
  **90.68%**；正常完成457、失败0、resolved 253。资源窗口为P/D Forward
  **79.41%/96.86%**、P/D KV **10.23%/34.79%**、Prefix复用率
  **96.48%**。整机内存allocation解决了运行中断，但未达到吞吐通过线。
- v5分段诊断：T0–T50 Decode为2,183.11 tok/s；T150–T250降至
  1,503.30 tok/s。服务日志在该窗口捕获两次P→D Direct DMA长尾，分别为
  **61.315s**和**56.256s**；单个`p2d_direct_lanes=1`被占用时，其余两个
  Decode组的P→D供给发生队头阻塞。全程仅11次D→P Host restore，因此本轮
  差距不能归因于30秒窗口后的D→P Host恢复。
- v6验证：其余配置完全不变，仅将P→D Direct lane从1增至3，与三个Decode
  组一一匹配，验证隔离P→D长尾能否使T450 Decode达到HiCache的
  1,861.40 tok/s。
- v6 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft30-p2dl3-r6`
- v6控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft30-p2dl3-r6-20261016/state.json`
- v6结果：T450 **3,029.47s**，T450 Decode **1,778.64 tok/s**，为HiCache的
  **95.55%**；相对v5提升**5.37%**，但仍低于通过线**4.45%**。正常完成454、
  失败0、resolved 263。资源窗口为P/D Forward **79.39%/98.55%**、P/D KV
  **9.79%/35.11%**、Prefix复用率**96.44%**。
- v6诊断：P→D改为3 lanes后不再出现50–60秒DMA长尾，说明该修复有效。
  剩余瓶颈是P侧恢复准入过细：新方法P队列均值仅**0.07**、Prefill请求率
  **8.42 req/s**，HiCache对应**24.86**和**9.31 req/s**；新方法P Forward
  更高但Prefill compute token率低约14.4%，表现为小批次效率损失。
- v7验证：保持P→D Direct=3及其他配置不变，仅将D→P Direct lane从2增至4，
  让多个恢复请求并发进入P，验证是否能恢复Prefill批处理规模并越过HiCache线。
- v7 run：
  `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35a3b-c128-d2pfix-ft30-p2dl3-d2pl4-r7`
- v7控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-35b-a3b-d2p-restore-ft30-p2dl3-d2pl4-r7-20261016/state.json`
- v7结果：T450 **3,047.91s**，T450 Decode **1,805.69 tok/s**，为HiCache
  1,861.40 tok/s的**97.01%**，仍低**2.99%**；相对v6提升**1.52%**。
  T450时正常完成450、失败0、resolved 269。资源窗口为P/D Forward
  **78.42%/98.71%**、P/D KV **7.59%/36.19%**、Prefix复用率
  **96.43%**。
- v7结论：D→P Direct lane从2增至4只有小幅收益，且P Forward没有上升，
  说明剩余差距已不是单纯增加D→P恢复并发就能消除。1P:3D参数扫描到此记录；
  后续使用相同`fast-tool=30s`、D→P/P→D Direct=4/3及512/256GiB
  Host总量，在独立2P:2D矩阵中验证增加P算力是否能越过HiCache。
