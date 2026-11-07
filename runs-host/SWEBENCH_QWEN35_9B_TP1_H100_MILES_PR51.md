# SWE-bench Verified：Qwen3.5-9B TP=1 单节点 H100（Miles PR51）

> **状态：完成（12/12）**  
> 完成时间：2026-10-14 08:35:59 UTC  
> 正式并发：`128 / 192 / 256`；每档4种方法，共12组。

## 结论

- 当前新方法相对No-reverse PD的T450加速为 **3.08× / 3.14× / 2.58×**；
  相对原生HiCache/Mooncake为 **1.53× / 1.85× / 2.02×**。
- 新方法尚未超过Colocated：以`对照方法 T450 / 新方法 T450`计算，三档分别为
  `0.85× / 0.82× / 0.77×`。Colocated仍是本矩阵中T450最短的基线。
- 新方法将P卡Forward占比稳定在 **28.15%–32.19%**；No-reverse PD为
  **99.09%–99.74%**，HiCache随并发从 **64.57%** 上升到 **90.64%**。
  这一利用率结构与减少多轮重复Prefill的设计目标一致，主要瓶颈已转移到Decode。
- 新方法D卡Forward占比为 **97.84%–99.29%**，窗口Decode吞吐为
  **3,281.63–3,771.83 tok/s**；相对No-reverse为 **2.47×–3.03×**。
- Resolved在`140–164/500`之间波动，未显示某种方法有稳定正确率优势；性能比较以
  固定500题、相同harness下的T450和Decode吞吐为主。

### T450加速汇总

> 加速比定义为`对照方法 T450 / 当前新方法 T450`，大于1表示新方法更快。

| 并发 | 相对Colocated | 相对No-reverse PD | 相对HiCache/Mooncake |
|---:|---:|---:|---:|
| 128 | 0.85× | **3.08×** | **1.53×** |
| 192 | 0.82× | **3.14×** | **1.85×** |
| 256 | 0.77× | **2.58×** | **2.02×** |

## 实验配置

| 项目 | 配置 |
|---|---|
| GPU | 单节点8×H100 80GB |
| 模型 | Qwen3.5-9B，TP=1 |
| 数据 | SWE-bench Verified固定500题 |
| 并发 | 128、192、256 |
| PD拓扑 | 当前新方法、No-reverse、HiCache均为2P:6D |
| Colocated | 8个TP1副本 |
| HBM配置 | 所有模型worker统一使用`mem_fraction_static=0.8` |
| 当前新方法Host | D→P总量512GiB；P→D总量256GiB |
| HiCache Host | Attention 33GB/worker；Mamba 24GB/worker；Mooncake共享segment 256GB |
| 正式节点 | 主要结果：`node2700` / job `25642154`；c256 HiCache r22：`node2800` / job `25734937` |

Miles PR51 harness保持与A100 Qwen3.5-9B实验一致：

- `action_protocol=fenced_shell`，`command_contract=miles_pr51`；
- 最多64轮，每轮最多8,192 token，历史窗口12，开启thinking；
- 累计response上限524,288 token，source-order，seed 2026；
- temperature 0.6、top-p 0.95、top-k 20；
- 使用相同500个SWE镜像和inline verifier；H100侧仅容器运行时改为对应SIF。

HiCache使用SGLang `0.5.10.post1` baseline源码、run-local `qwen35_mooncake`
compatibility layer和Mooncake `0.3.12.post1`。compatibility layer通过
`sitecustomize`补齐`HybridLinearKVPool`的KV+Mamba offload/restore，不修改
baseline安装。主要依赖为Torch `2.9.1`、`sglang-kernel 0.4.1`、FlashInfer
`0.6.7.post3`和Transformers `5.3.0`。

## 指标口径

- **T450**：从workload测量起点到第450个terminal result的墙钟时间；正常完成和
  workload失败均计入terminal result。
- **T450前Decode吞吐**：同一测量起点到T450边界之间，各Decode endpoint的
  `sglang_realtime_tokens_total{mode="decode"}` 单调counter增量之和除以T450。
  counter按endpoint分别做差后再求和，边界位于两个scrape之间时线性插值。
- T450时仍在运行的请求已经生成的Decode token会计入该指标，因此比只累计前450个
  已结束任务的输出更能反映实际服务能力。
- 全程Prefill/Decode平均吞吐包含T450之后的有限任务drain和verifier长尾，仅作辅助；
  Forward表使用第50至第450个terminal result之间的窗口。

<!-- AUTOPILOT_RESULTS_START -->
## 正式结果

> 最后检查：2026-10-14 08:35:59 UTC。
> 状态：`complete`；当前完整产物 **12/12**；执行顺序为c128/c192非HiCache，最后执行c128/c192/c256 HiCache。
> 新增9组实验均已完成；加上复用的c256三组，共12组。

### 核心指标

| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |
|---:|---|---:|---:|---:|---:|---:|
| 128 | 当前新方法（2P:6D） | 500/0 | 147/500 | **1,624.47** | **3,842.76** | 6.242M |
| 128 | Colocated（8副本） | 500/0 | 155/500 | **1,382.97** | **4,739.09** | 6.554M |
| 128 | No-reverse PD（2P:6D） | 500/0 | 164/500 | **5,010.08** | **1,345.89** | 6.743M |
| 128 | 原生 HiCache/Mooncake（2P:6D） | 500/0 | 151/500 | **2,477.91** | **2,721.46** | 6.744M |
| 192 | 当前新方法（2P:6D） | 500/0 | 149/500 | **1,633.32** | **4,104.91** | 6.705M |
| 192 | Colocated（8副本） | 500/0 | 161/500 | **1,336.14** | **5,219.92** | 6.975M |
| 192 | No-reverse PD（2P:6D） | 499/1 | 148/500 | **5,126.06** | **1,353.72** | 6.939M |
| 192 | 原生 HiCache/Mooncake（2P:6D） | 499/1 | 153/500 | **3,023.81** | **2,125.21** | 6.426M |
| 256 | 当前新方法（2P:6D） | 499/1 | 159/500 | **1,885.19** | **3,506.66** | 6.611M |
| 256 | Colocated（8副本） | 500/0 | 146/500 | **1,455.06** | **4,515.91** | 6.571M |
| 256 | No-reverse PD（2P:6D） | 499/1 | 157/500 | **4,862.30** | **1,420.34** | 6.906M |
| 256 | 原生 HiCache/Mooncake（2P:6D） | 499/1 | 140/500 | **3,816.18** | **1,735.61** | 6.623M |

### 全程与辅助指标

| 并发 | 方法 | T50 (s) | T250 (s) | T500 (s) | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 当前新方法（2P:6D） | 289.62 | 951.04 | 2,205.06 | 0.29966 | 4,992.28 | 2,990.44 | 5 | — |
| 128 | Colocated（8副本） | 219.95 | 685.20 | 3,969.04 | 0.34393 | 4,619.79 | 1,695.73 | 11 | — |
| 128 | No-reverse PD（2P:6D） | 796.14 | 3,106.54 | 5,537.90 | 0.09492 | 50,859.60 | 1,262.99 | 116 | — |
| 128 | 原生 HiCache/Mooncake（2P:6D） | 409.72 | 1,488.09 | 2,888.91 | 0.19341 | 24,417.35 | 2,439.95 | 113 | — |
| 192 | 当前新方法（2P:6D） | 266.33 | 945.98 | 2,218.09 | 0.29261 | 5,048.54 | 3,125.46 | 4 | — |
| 192 | Colocated（8副本） | 263.53 | 771.34 | 1,742.34 | 0.37292 | 15,474.63 | 4,036.19 | 10 | — |
| 192 | No-reverse PD（2P:6D） | 677.00 | 3,003.40 | 8,162.02 | 0.08991 | 35,263.39 | 872.66 | 184 | env=1, failed=1 |
| 192 | 原生 HiCache/Mooncake（2P:6D） | 349.80 | 2,141.82 | 3,915.56 | 0.14959 | 31,676.32 | 1,691.24 | 180 | env=1, failed=1 |
| 256 | 当前新方法（2P:6D） | 410.89 | 1,211.30 | 2,384.06 | 0.27132 | 4,718.36 | 2,844.84 | 32 | env=1, failed=1 |
| 256 | Colocated（8副本） | 314.09 | 852.14 | 1,756.10 | 0.35058 | 18,941.14 | 3,742.09 | 20 | — |
| 256 | No-reverse PD（2P:6D） | 673.90 | 3,321.35 | 5,146.41 | 0.09550 | 53,283.44 | 1,377.23 | 243 | env=1, failed=1 |
| 256 | 原生 HiCache/Mooncake（2P:6D） | 365.41 | 2,775.20 | 4,342.74 | 0.11592 | 39,800.87 | 1,556.01 | 247 | env=1, failed=1 |

### T50–T450 P/D Forward占比与窗口吞吐

下表直接在每组第50至第450个terminal result的绝对时间边界上，对各scheduler累计
counter做差。`token/GPU-forward-s`为角色总token/s除以该角色GPU数和每卡Forward
占比，表示GPU实际执行该角色Forward期间的等效吞吐，不是墙钟吞吐。

| 并发 | 方法 | 窗口(s) | Agent/s | P token/s | P Forward/卡 | P token/GPU-forward-s | D token/s | D Forward/卡 | D token/GPU-forward-s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 当前新方法（2P:6D） | 1,334.85 | 0.29966 | 5,820.12 | 31.82% | 9,145.82 | 3,644.87 | 99.29% | 611.85 |
| 128 | Colocated（8副本） | 1,163.02 | 0.34393 | 11,330.49 | 11.11% | 12,743.28 | 4,459.97 | 87.52% | 637.02 |
| 128 | No-reverse PD（2P:6D） | 4,213.94 | 0.09492 | 53,884.40 | 99.09% | 27,189.88 | 1,280.92 | 85.99% | 248.28 |
| 128 | 原生 HiCache/Mooncake（2P:6D） | 2,068.19 | 0.19341 | 28,886.78 | 64.57% | 22,367.40 | 2,671.13 | 94.46% | 471.32 |
| 192 | 当前新方法（2P:6D） | 1,366.99 | 0.29261 | 5,593.00 | 32.19% | 8,688.75 | 3,771.83 | 99.22% | 633.58 |
| 192 | Colocated（8副本） | 1,072.62 | 0.37292 | 15,145.88 | 12.67% | 14,938.20 | 4,712.93 | 82.25% | 716.22 |
| 192 | No-reverse PD（2P:6D） | 4,449.06 | 0.08991 | 54,033.04 | 99.67% | 27,106.28 | 1,231.87 | 83.77% | 245.10 |
| 192 | 原生 HiCache/Mooncake（2P:6D） | 2,674.01 | 0.14959 | 42,099.76 | 85.21% | 24,703.04 | 1,952.14 | 90.19% | 360.75 |
| 256 | 当前新方法（2P:6D） | 1,474.30 | 0.27132 | 4,860.13 | 28.15% | 8,631.71 | 3,281.63 | 97.84% | 559.03 |
| 256 | Colocated（8副本） | 1,140.97 | 0.35058 | 16,879.77 | 12.64% | 16,698.15 | 3,920.94 | 63.30% | 774.25 |
| 256 | No-reverse PD（2P:6D） | 4,188.41 | 0.09550 | 54,154.98 | 99.74% | 27,147.60 | 1,264.63 | 84.16% | 250.43 |
| 256 | 原生 HiCache/Mooncake（2P:6D） | 3,450.77 | 0.11592 | 46,318.63 | 90.64% | 25,550.48 | 1,553.86 | 86.84% | 298.24 |

Colocated的P/D数据来自同一组8个引擎，其他方法分别来自2张P卡与6张D卡；因此
Colocated的两个Forward占比可按同卡时间构成理解，PD方法不能跨角色直接相加。

### 原始产物

- c128 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-agentic-2p6d-swe500-miles-stock256-c128-20261012-r15`
- c128 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-colocated8-swe500-miles-stock256-c128-20261012-r15`
- c128 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-native-2p6d-swe500-miles-stock256-c128-20261012-r15`
- c128 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-hicache-2p6d-swe500-miles-stock256-c128-20261012-r17`
- c192 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-agentic-2p6d-swe500-miles-stock256-c192-20261012-r16`
- c192 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-colocated8-swe500-miles-stock256-c192-20261012-r16`
- c192 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-native-2p6d-swe500-miles-stock256-c192-20261012-r16`
- c192 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-hicache-2p6d-swe500-miles-stock256-c192-20261012-r17`
- c256 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-agentic-2p6d-swe500-miles-c256-20261010-r2`
- c256 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-colocated8-swe500-miles-c256-20261010-r3`
- c256 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-native-oneway-2p6d-swe500-miles-c256-20261010-r2`
- c256 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-9b-h100-hicache-2p6d-swe500-miles-stock256-c256-20261012-r22`
<!-- AUTOPILOT_RESULTS_END -->

## 有效性与运行审计

### 正式结果判定

- 12组均包含500个唯一任务和500个terminal result；`完成/失败`中的失败也计入
  T450边界，但不计入正常完成数。
- c192 No-reverse、c192 HiCache、c256新方法、c256 No-reverse和c256 HiCache各有
  1个环境失败；其余组为500/0。表中Resolved分母统一保持500。
- 方法性能结论以核心表中的T450和T450前Decode吞吐为准；Forward表解释各角色
  利用率及其活跃计算期间的等效吞吐。

### c256 HiCache修复链

1. 早期r6在8/500首次Host eviction时触发`parent does not have child key`。根因是旧版
   `HiMambaRadixCache`把仍有子节点的内部节点登记为可淘汰叶子；移植上游
   stale-eviction条件修复，未改变Mooncake或容量参数。
2. r7通过11轮Mooncake eviction后，在34/500触发DeepGEMM JIT
   `pipe != nullptr`；该run判为无效。
3. 后续诊断确认stock Mooncake master的`BatchEvict`每轮临时创建16个线程，在高Host
   提交量下会以`std::system_error: Resource temporarily unavailable`退出。最终master
   改为固定16-worker池，淘汰对象、分片扫描和淘汰语义保持不变。
4. r21中修复版master稳定完成84/84轮淘汰、累计1.49TB，但Prefill在75/500遇到迟到
   Triton JIT，LLVM临时线程池创建失败。r22设置`MLIR_DISABLE_MULTITHREADING=1`，
   仅将首次编译串行化；模型、kernel语义、2P:6D拓扑、Host容量和Mooncake机制不变。
5. r22最终得到499个正常完成和1个环境失败，T450边界完整，且运行结束后GPU、端口、
   Apptainer和`squashfuse_ll`全部归零。

MLIR串行编译可能增加测量窗口内首次遇到新shape时的JIT停顿，因此对c256 HiCache至多
构成保守劣势，不会凭空提高其稳态吞吐。

### 并发与资源门禁

- c394/c500诊断发现持久Apptainer sandbox会放大节点线程和虚拟内存压力，因此正式矩阵
  收敛为c128/c192/c256；历史失败目录不纳入结果。
- 节点使用`vm.overcommit_memory=0`，CommitLimit约为2TiB物理内存的一半。约712GB
  Host层预留会自然推高`Committed_AS`，所以它只做观测而不是硬门禁。
- 正式门禁为：至少256GiB可用物理内存、服务拓扑健康、100,000线程失控保险，以及每组
  结束后GPU进程、矩阵端口、Apptainer和FUSE helper连续两次归零；清理失败则取消并
  重建allocation，不继续下一组。
