# SWE-bench Verified · Qwen3.8-27B TP=2 · 单节点H100 · 2P:2D矩阵

> 更新：2026-10-20。初始9组矩阵在 `node2700` 上串行执行；另在
> `node1800/node2701` 补测更新版官方SGLang的c128/c192 HiCache，结果单独列出，
> 不覆盖旧baseline容量失败记录。

## 实验范围

- 模型：Qwen3.8-27B，TP=2，BF16，context 131,072。
- 数据：SWE-bench Verified固定500题，source-order，seed 2026。
- Harness：Qwen3.8-27B既有验证路径 `openenv_structured_tool`；最多64轮、每轮最多
  8,192 token，temperature 0.6、top-p 0.95、top-k 20。
- Sandbox：四种方法统一使用Apptainer原生稀疏overlay（每题上限4GiB）；模型、题目、
  verifier和并发口径不变。该模式已在node2700通过完整启动、持久命令、patch capture
  和inline verifier smoke，避免`host_bind`为每个并发题完整复制`/testbed`。
- 并发：96、128、192；每档依次运行当前新方法、No-reverse PD、原生
  HiCache/Mooncake。Colocated布局和既有单节点TP2实验相同，按要求不重复测试，
  因此不计入本轮9组新实验。
- 单节点布局：`P0=GPU0-1`、`D0=GPU2-3`、`P1=GPU4-5`、`D1=GPU6-7`；
  Colocated使用4个TP2完整副本。
- 当前新方法：D→P Host总量512GiB（每个D、每rank 128GiB），P→D Host总量
  256GiB（每个P、每rank 64GiB）；原生HiCache使用baseline SGLang + Mooncake
  原生机制和256GB共享segment。
- Colocated和No-reverse使用从`b1ef6ef00`隔离出的干净baseline，只补入两项既有
  Qwen3.8-27B正确性修复：fallback GDN的`a/b.contiguous()`与原生Decode Mamba
  slot原子admission；不包含新方法、D→P或Host slow path代码。HiCache仍使用原生
  Mooncake目录与机制。

## 核心指标口径

- **T450**：从workload测量起点到第450个terminal result的墙钟时间。
- 自2026-10-17起，后续各组达到第450条terminal result并持久化T450计数器窗口后
  立即停止；不再等待最后50题长尾。表中的正确率分母因此以该run实际目标（450或
  已完成历史run的500）为准，T500对主动停止的run留空。
- **T450前Decode吞吐**：同一时间段内，各Decode endpoint的
  `sglang_realtime_tokens_total{mode="decode"}` counter增量之和除以T450；各endpoint
  先独立做差再求和，边界位于两次scrape之间时线性插值。
- 全程吞吐包含T450后的有限任务drain和verifier长尾，仅作为辅助指标。

## 串行与清理约束

执行顺序为 `c96 → c128 → c192`，每档均按“当前新方法 → No-reverse →
HiCache”运行。每组切换前必须确认GPU进程、矩阵端口、Apptainer、
`squashfuse_ll`和`faked-sysv`全部归零；未清理干净时不得启动下一组。

### 历史无效试跑

2026-10-13的c128 Agentic r1/r2/r3分别仅留下0、1、23条terminal记录，均未达到
T450且不属于本次c96/c128/c192矩阵；这些目录只保留故障审计，不进入正式表格。
2026-10-16的c96 Agentic r1在prepare参数校验阶段退出，未启动GPU服务或workload；
随后r2使用旧`host_bind`工作区，在225/500时因1800GiB job cgroup峰值达到约
1782GiB而触发256GiB可用内存门禁；该run无environment error但未达到T450，保留为
无效审计记录。正式矩阵改用四种方法统一的稀疏overlay，从`overlay-r1`重新开始。
首次c96 HiCache `overlay-r1`仅启动到P0参数校验：b1ef6 baseline不接受新版本CLI
`--mamba-radix-cache-strategy`。该次未启动workload、不计入结果；正式HiCache沿用此前
成功run的原生参数名`--mamba-scheduler-strategy`，从`overlay-r2`开始。
`overlay-r2`在266/500前无environment error，但verifier/overlay并发峰值令
MemAvailable短时降至约166–176GiB；job cgroup仅发生回收（`oom=0`、`oom_kill=0`）且
一分钟内恢复。旧256GiB单点门禁因此将其提前判为无效。本矩阵改用128GiB紧急余量，
保留OOM与零残留检查，HiCache从`overlay-r3`完整重测。

c128当前新方法`overlay-r1`在335条terminal记录时由`decode-0`触发
`Can not alloc mamba cache`；该轮后续记录均为连接失败，已标记无效。根因是
通用launcher的0.5 Mamba/HBM分配只提供169个slot，而每个活跃请求需要3个slot，
不足以覆盖c128在单个Decode组上的峰值。c128/c192从`overlay-r2`开始仅在本矩阵
manifest中将Mamba full-memory ratio提高到0.9；不修改共享launcher默认值，也不影响
其他正在运行的实验。

c128原生HiCache `overlay-r3`保持baseline代码及全部参数不变，在132条terminal记录
时因单个Decode组的原生Mamba pool达到240/240而退出；4条已正常完成，随后128条为
服务退出后的502。该轮记录为原生baseline容量失败，不调显存比例、不改分流或代码，
也不重跑调参版本；矩阵直接继续c192。

c192原生HiCache `overlay-r3`同样保持baseline代码及参数不变；两个Decode组分别在
原生Mamba pool达到240/240后退出，随后Prefill出现NIXL remote disconnect，workload
产生364条502/503环境错误且没有正常terminal。该轮未达到T450，作为原生baseline
容量失败封存，不把服务退出后的错误记录计作性能结果。

## 更新版官方SGLang HiCache补测（2026-10-19）

旧`b1ef6` baseline的c128/c192 HiCache均因原生Mamba pool容量耗尽而失败。为区分
“方法本身不可运行”和旧版本实现缺口，另用SGLang v0.5.23、commit `617c8b67`
在独立环境重测。官方source tree保持不变；由于该版本的
`DecodeKVCacheOffloadManager`尚不支持Qwen3.8的`HybridLinearKVPool`，GPU服务通过
run-local `qwen35_mooncake_v0523`兼容桥接入v0.5.23原生
`build_hybrid_mamba_stack`，补齐KV+Mamba Host保存、恢复和释放生命周期。该桥接不含
当前新方法的D→P/P→D调度代码，因此以下结果作为“更新版官方SGLang兼容补测”单独
报告，不回填为初始9组矩阵中的旧baseline结果。

两组保持Qwen3.8-27B、TP=2、单节点2P:2D、`mem_fraction_static=0.8`、
Mamba full-memory ratio 0.9、每rank 32GiB Attention HiCache + 32GiB Mamba
HiCache，以及256GiB Mooncake segment；仅并发分别为128和192。

### 核心结果

| 并发 | 完成/失败 | Resolved（全500题） | Resolved@T450 | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 500/0 | 249/500（49.80%） | 216/450（48.00%） | **12,613.49** | **870.53** | 10.980M |
| 192 | 500/0 | 245/500（49.00%） | 215/450（47.78%） | **13,694.83** | **814.87** | 11.159M |

### 全程与T50–T450辅助指标

| 并发 | T50 | T250 | T500 | T50–T450窗口 | Agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 751.26 | 5,432.57 | 16,560.34 | 11,862.24 | 0.03372 | 8,756.40 | 694.43 | 60 |
| 192 | 649.77 | 5,654.25 | 15,017.87 | 13,045.07 | 0.03066 | 10,815.08 | 768.78 | 86 |

### T50–T450 P/D Forward、复用与缓存压力

Forward仍按TP=2先做TP组均值、再除以2，表示每张物理GPU的墙钟占比；P和D位于
不同GPU，不能相加。有效复用率为
`P cache / (P cache + P compute)`。缓存与队列值均为窗口内每个TP组的时间均值。

| 并发 | P compute tok/s | D tok/s | P Forward/GPU | D Forward/GPU | 有效复用率 | P queue | D running | D prealloc / transfer | Attention KV P/D | Mamba P/D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 11,184.27 | 762.23 | 52.40% | 99.36% | 53.99% | 5.53 | 10.25 | 41.85 / 6.59 | 3.95% / 72.09% | 2.04% / 7.04% |
| 192 | 11,559.72 | 702.76 | 52.66% | 97.96% | 49.37% | 6.05 | 9.27 | 60.93 / 8.02 | 3.88% / 75.89% | 1.99% / 7.23% |

两组均有500条唯一terminal记录、500条request记录、`summary.json`以及覆盖完整
T50–T450窗口的逐endpoint metrics；environment error均为0。workload完成后控制器曾
被成功启动前残留的旧`INVALID.md`阻止完成判定，直至12小时allocation到期才报警；
该标记的时间早于正式measurement start，现保留为`INVALID.prestart-stale.md`审计，
最终结果封存在各run的`COMPLETED_RESULT.json`。这属于控制器收尾误判，不影响上述
已落盘测量数据。

原始目录：

- c128：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-node1800-1n-2p2d-hicache-mooncake-upstream617c-v0523compat-swe500-c128-20261019-r6`
- c192：`/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-node2701-1n-2p2d-hicache-mooncake-upstream617c-v0523compat-swe500-c192-20261019-r6`

<!-- AUTOPILOT_RESULTS_START -->
## 自动运行进度与结果

> 最后检查：2026-10-18 20:18:23 UTC。
> 状态：`complete`；有效结果 **7/9**。
> 9组均已有终态：7组取得有效结果，c128/c192原生HiCache因baseline Mamba容量失败，
> 未取得T450。

### 核心指标

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 96 | 当前新方法（2P:2D） | 499/1 | 255/500 | **6,683.72** | **1,594.90** | 10.660M |
| 96 | No-reverse PD（2P:2D） | 500/0 | 253/500 | **18,178.07** | **609.92** | 11.087M |
| 96 | 原生 HiCache/Mooncake（2P:2D） | 450/0 | 215/450 | **11,004.90** | **940.87** | 10.354M |
| 128 | 当前新方法（2P:2D） | 445/5 | 205/450 | **6,964.59** | **1,464.32** | 10.198M |
| 128 | No-reverse PD（2P:2D） | 450/0 | 251/450 | **22,649.70** | **595.29** | 13.483M |
| 128 | 原生 HiCache/Mooncake（2P:2D） | 失败（4/132） | — | — | — | — |
| 192 | 当前新方法（2P:2D） | 448/2 | 216/450 | **7,279.01** | **1,423.79** | 10.364M |
| 192 | No-reverse PD（2P:2D） | 449/1 | 243/450 | **26,728.50** | **635.84** | 16.995M |
| 192 | 原生 HiCache/Mooncake（2P:2D） | 失败（0/364） | — | — | — | — |

### 全程与辅助指标

| 并发 | 方法 | T50 | T250 | T500 | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 96 | 当前新方法（2P:2D） | 578.40 | 2,927.11 | 8,579.62 | 0.06552 | 1,986.02 | 1,297.11 | 0 | env=1, failed=1 |
| 96 | No-reverse PD（2P:2D） | 1,272.02 | 9,499.33 | 20,773.27 | 0.02366 | 17,844.00 | 556.29 | 75 | — |
| 96 | 原生 HiCache/Mooncake（2P:2D） | 769.38 | 4,620.03 | — | 0.03908 | 11,637.91 | 940.87 | 73 | — |
| 128 | 当前新方法（2P:2D） | 601.63 | 3,191.44 | — | 0.06286 | 2,225.76 | 1,464.32 | 0 | env=5, failed=5 |
| 128 | No-reverse PD（2P:2D） | 956.42 | 9,406.93 | — | 0.01844 | 18,332.21 | 595.29 | 66 | — |
| 128 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — | — | baseline Mamba pool 240/240；4正常，128×502 |
| 192 | 当前新方法（2P:2D） | 600.39 | 2,910.20 | — | 0.05989 | 2,192.81 | 1,423.79 | 3 | env=1, failed=2 |
| 192 | No-reverse PD（2P:2D） | 774.59 | 8,767.64 | — | 0.01541 | 18,174.24 | 635.84 | 70 | failed=1 |
| 192 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — | — | native baseline Mamba pool exhausted at 240/240 in both Decode groups; 0 normal and 364 environment-error terminals; no valid T450 result |

原始目录：

- c96 当前新方法（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-agentic-swe500-c96-20261016-overlay-r1`
- c96 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-native-oneway-swe500-c96-20261016-overlay-r1`
- c96 原生 HiCache/Mooncake（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-hicache-mooncake-swe500-c96-20261016-overlay-r3`
- c128 当前新方法（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-agentic-swe500-c128-20261016-overlay-r2`
- c128 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-native-oneway-swe500-c128-20261016-overlay-r1`
- c192 当前新方法（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-agentic-swe500-c192-20261016-overlay-r2`
- c192 No-reverse PD（2P:2D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-native-oneway-swe500-c192-20261016-overlay-r1`
- c128 原生 HiCache/Mooncake（2P:2D） （失败审计）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-hicache-mooncake-swe500-c128-20261016-overlay-r3`
- c192 原生 HiCache/Mooncake（2P:2D） （失败审计）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q38-27b-h100-1n-2p2d-hicache-mooncake-swe500-c192-20261016-overlay-r3`
<!-- AUTOPILOT_RESULTS_END -->
<!-- T50_T450_DETAILS_START -->
## T50–T450 P/D Forward 与吞吐

下表统一取第50条至第450条terminal result之间的主体窗口。Prefill/Decode tok/s为scheduler实际计算量。

Forward计数器包含两个TP rank，因此先按TP组求均值，再除以TP=2，得到每张物理GPU的墙钟占比。PD的P/D位于不同GPU，二者不能相加；Colocated的Extend/Decode位于同一批GPU，可以相加理解计算时间构成。

| 并发 | 方法 | 窗口(s) | Agent/s | P compute tok/s | D tok/s | P/Extend Forward/GPU | D/Decode Forward/GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 96 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 96 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |
| 128 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 128 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 128 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |
| 192 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 192 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 192 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |

## T50–T450 复用率与缓存/队列

有效复用率按 `P cache / (P cache + P compute)` 计算，等价于 `1 - 实际Prefill计算量 / 逻辑Prompt量`。No-reverse关闭复用且不导出 `prefill_cache` counter，因此按精确的0%记录。

| 并发 | 方法 | 有效复用率 | P queue | D running | D prealloc / transfer | Attention KV P/D | Mamba P/D |
|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 96 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 96 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |
| 128 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 128 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 128 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |
| 192 | 当前新方法（2P:2D） | — | — | — | — | — | — |
| 192 | No-reverse PD（2P:2D） | — | — | — | — | — | — |
| 192 | 原生 HiCache/Mooncake（2P:2D） | — | — | — | — | — | — |

Attention KV/Mamba为各TP组活跃、不可驱逐缓存池的时间均值，不是整卡HBM占比；queue、running、prealloc、transfer均为每个TP组的时间均值。空白行表示该run尚未达到T450或时序metrics尚不完整。
<!-- T50_T450_DETAILS_END -->
