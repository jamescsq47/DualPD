# SWE-bench Verified · Qwen3.5-9B TP=1 · 单节点 H100 矩阵

> **作废说明（2026-10-10）：** 本文12组误用了Qwen3.5-27B的
> `openai_tools / structured-tool` harness，不能与A100 Qwen3.5-9B结果比较，
> 也不能用于判断四种方法的相对性能。数据仅保留供故障审计。正式重测使用
> `miles_pr51 / fenced_shell`，记录于
> `SWEBENCH_QWEN35_9B_TP1_H100_MILES_PR51.md`。

> 创建：2026-10-09。计划12组；结果由自动控制器逐组原子写入。

## 实验范围

本矩阵在唯一允许使用的 `node2700` 上比较 Qwen3.5-9B TP=1 完成
SWE-bench Verified 500道不同题的性能。并发为 `c256 / c394 / c500`，每个并发按
以下顺序运行四种方法：当前新方法、Colocated、No-reverse PD、原生
HiCache/Mooncake PD。第一指标为 **T450**；T500仅作为长尾收尾参考。

统一配置：

- 硬件：单节点8×NVIDIA H100 80GB；Slurm job `25279337`，只使用 `node2700`。
- 模型：`Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 数据：SWE-bench Verified 500，source-order，每题执行一次；Apptainer inline verifier。
- 采样：temperature 0.6、top-p 0.95、top-k 20、seed 2026。
- 上下文131,072；单轮最多8,192 token；最多64轮；Prefill chunk/batch 8,192。
- 所有方法 `mem_fraction_static=0.8`、`mamba_full_memory_ratio=0.5`。
- 当前新方法：TP1 2P:6D，P在GPU 0/4，D在GPU 1/2/3/5/6/7；D→P Host
  在6个D源rank上按 `86+86+85+85+85+85 = 512 GiB` 分配，P→D Host在2个P源rank上按
  `128+128 = 256 GiB` 分配；原生
  HiCache/Mooncake关闭。
- Colocated：8个TP1完整副本，不启用PD、HiCache或Mooncake。
- No-reverse：原生P→D NIXL，D→P关闭，下一轮重新Prefill；2P:6D。
- 原生HiCache/Mooncake：2P:6D，保留stock SGLang HiCache/Mooncake机制和
  512GiB共享Mooncake segment，不套用当前新方法的方向拆分Host语义；H100本地
  Attention/Mamba host tier分别为33/17 GB每rank（均严格大于对应设备池）。

T50/T250/T450/T500均从 workload 测量起点计，包含正常结束与失败题。每组必须形成
500条唯一 episode progress 和完整 summary 才会自动进入下一组；切换前确认该run的
GPU进程与run-scoped进程全部清零，避免端口或残留进程污染下一组。

<!-- AUTOPILOT_RESULTS_START -->
## 自动运行进度与结果

> 最后检查：2026-10-10 01:40:41 UTC。
> 状态：`complete`；完整产物 **12/12**。
> 12组均已完成。

| 并发 | 方法 | 完成/失败 | Resolved | T50 (s) | T250 (s) | **T450 (s)** | T500 (s) | T50–T450 agent/s | Prefill tok/s | Decode tok/s | Max queue | 异常 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 当前新方法（2P:6D） | 500/0 | 56/500 | 217.79 | 486.24 | **1,141.79** | 1,323.66 | 0.43290 | 3,306.09 | 599.78 | 115 | — |
| 256 | Colocated（8副本） | 500/0 | 49/500 | 57.84 | 280.49 | **839.09** | 1,100.80 | 0.51200 | 3,717.84 | 692.35 | 0 | — |
| 256 | No-reverse PD（2P:6D） | 500/0 | 60/500 | 79.22 | 392.87 | **881.65** | 1,082.01 | 0.49849 | 24,618.43 | 736.26 | 223 | — |
| 256 | 原生 HiCache/Mooncake（2P:6D） | 500/0 | 37/500 | 67.96 | 221.23 | **741.27** | 2,644.34 | 0.59408 | 2,474.16 | 300.79 | 121 | — |
| 394 | 当前新方法（2P:6D） | 500/0 | 48/500 | 56.24 | 271.17 | **815.09** | 1,141.47 | 0.52711 | 3,440.28 | 636.65 | 7 | — |
| 394 | Colocated（8副本） | 500/0 | 49/500 | 55.66 | 273.04 | **834.67** | 1,124.18 | 0.51348 | 3,414.49 | 658.38 | 0 | — |
| 394 | No-reverse PD（2P:6D） | 500/0 | 57/500 | 59.17 | 292.95 | **690.34** | 1,085.83 | 0.63375 | 24,149.00 | 688.82 | 299 | — |
| 394 | 原生 HiCache/Mooncake（2P:6D） | 500/0 | 60/500 | 58.66 | 247.78 | **769.68** | 2,570.81 | 0.56257 | 2,753.85 | 379.04 | 122 | — |
| 500 | 当前新方法（2P:6D） | 500/0 | 52/500 | 60.55 | 279.05 | **807.85** | 1,076.04 | 0.53526 | 3,640.49 | 692.07 | 8 | — |
| 500 | Colocated（8副本） | 500/0 | 56/500 | 54.25 | 260.57 | **802.73** | 1,126.11 | 0.53441 | 3,700.24 | 660.61 | 3 | — |
| 500 | No-reverse PD（2P:6D） | 500/0 | 62/500 | 62.12 | 317.86 | **804.72** | 2,526.69 | 0.53864 | 10,524.58 | 311.03 | 293 | — |
| 500 | 原生 HiCache/Mooncake（2P:6D） | 500/0 | 57/500 | 54.18 | 214.67 | **661.97** | 1,042.77 | 0.65812 | 6,579.05 | 853.42 | 122 | — |

原始目录：

- c256 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c256-20261009-r1`
- c256 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-colocated8-swe500-c256-20261009-r3`
- c256 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c256-20261009-r2`
- c256 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c256-20261009-r9`
- c394 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c394-20261009-r1`
- c394 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-colocated8-swe500-c394-20261009-r1`
- c394 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c394-20261009-r1`
- c394 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c394-20261009-r2`
- c500 当前新方法（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c500-20261009-r2`
- c500 Colocated（8副本）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-colocated8-swe500-c500-20261009-r1`
- c500 No-reverse PD（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c500-20261009-r1`
- c500 原生 HiCache/Mooncake（2P:6D）: `/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c500-20261009-r1`
<!-- AUTOPILOT_RESULTS_END -->

## T50–T450 P/D Forward 与窗口吞吐

自动主表中的Prefill/Decode token/s沿用workload `summary.json`的全程口径；下表则严格截取
T50–T450主体段，因此不会混入启动段和T450之后的verifier长尾。

下表按每组第50条至第450条episode回调的绝对时间边界，直接对scheduler累计计数器做差。
`token/GPU-forward-s` 为角色总 token/s 除以该角色GPU数和每卡Forward占比，表示每个GPU实际处于对应Forward期间的等效吞吐；它不是墙钟吞吐。

| 并发 | 方法 | 窗口(s) | Agent/s | P token/s | P Forward/卡 | P token/GPU-forward-s | D token/s | D Forward/卡 | D token/GPU-forward-s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 当前新方法 | 924.00 | 0.43290 | 2,886.09 | 16.35% | 8,825.35 | 577.42 | 33.86% | 284.18 |
| 256 | Colocated | 781.25 | 0.51200 | 3,837.67 | 5.79% | 8,279.12 | 774.19 | 25.67% | 376.99 |
| 256 | No-reverse PD | 802.43 | 0.49849 | 29,390.93 | 43.91% | 33,466.51 | 814.28 | 47.45% | 286.01 |
| 256 | 原生 HiCache/Mooncake | 673.31 | 0.59408 | 8,031.03 | 14.83% | 27,071.99 | 918.56 | 36.10% | 424.04 |
| 394 | 当前新方法 | 758.85 | 0.52711 | 4,119.37 | 12.52% | 16,447.09 | 791.97 | 35.39% | 372.93 |
| 394 | Colocated | 779.01 | 0.51348 | 3,837.67 | 3.19% | 15,045.73 | 771.65 | 20.80% | 463.71 |
| 394 | No-reverse PD | 631.17 | 0.63375 | 37,857.90 | 56.70% | 33,383.62 | 1,023.78 | 57.34% | 297.59 |
| 394 | 原生 HiCache/Mooncake | 711.02 | 0.56257 | 8,490.37 | 18.47% | 22,983.37 | 1,172.68 | 58.30% | 335.27 |
| 500 | 当前新方法 | 747.31 | 0.53526 | 3,964.82 | 12.08% | 16,404.52 | 797.14 | 32.55% | 408.18 |
| 500 | Colocated | 748.48 | 0.53441 | 4,186.69 | 3.38% | 15,476.73 | 797.16 | 21.31% | 467.68 |
| 500 | No-reverse PD | 742.61 | 0.53864 | 32,550.74 | 47.22% | 34,470.62 | 916.31 | 47.91% | 318.78 |
| 500 | 原生 HiCache/Mooncake | 607.79 | 0.65812 | 9,698.66 | 19.26% | 25,173.52 | 1,253.17 | 55.14% | 378.78 |

Colocated的P/D数据来自同一组8个引擎；其他三种方法的P和D分别来自独立的2张P卡与6张D卡。
因此Colocated的P/D Forward可直接相加理解单卡计算时间构成；PD方法则应分别看P卡和D卡，不能跨角色相加。

## T50–T450 资源均值

| 并发 | 方法 | P KV | P Mamba | P queue | P inflight | D KV | D Mamba | D running | D queue | D prealloc | D transfer | 最大采样间隔(s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 当前新方法 | 4.55% | 3.88% | 0.01 | 0.00 | 4.01% | 2.07% | 1.53 | 0.00 | 0.00 | 0.00 | 10.67 |
| 256 | Colocated | — | — | — | — | 1.20% | 1.65% | 1.19 | 0.00 | — | — | 13.43 |
| 256 | No-reverse PD | 1.10% | 0.61% | 28.89 | 1.19 | 7.47% | 11.22% | 1.35 | 0.00 | 0.00 | 10.09 | 3.96 |
| 256 | 原生 HiCache/Mooncake | 1.53% | 3.73% | 8.66 | 2.04 | 4.50% | 6.68% | 1.80 | 0.00 | 0.00 | 4.93 | 12.54 |
| 394 | 当前新方法 | 3.23% | 4.26% | 0.04 | 0.00 | 3.28% | 2.47% | 1.73 | 0.00 | 0.00 | 0.00 | 8.63 |
| 394 | Colocated | — | — | — | — | 0.83% | 1.47% | 1.08 | 0.00 | — | — | 7.09 |
| 394 | No-reverse PD | 2.43% | 0.67% | 39.28 | 1.30 | 9.89% | 14.90% | 1.78 | 0.00 | 0.01 | 13.67 | 4.32 |
| 394 | 原生 HiCache/Mooncake | 2.43% | 3.36% | 10.86 | 1.82 | 6.65% | 8.04% | 2.15 | 0.00 | 0.00 | 6.17 | 9.10 |
| 500 | 当前新方法 | 3.52% | 4.79% | 0.05 | 0.00 | 3.27% | 2.64% | 1.83 | 0.00 | 0.00 | 0.00 | 10.72 |
| 500 | Colocated | — | — | — | — | 0.85% | 1.59% | 1.13 | 0.00 | — | — | 15.81 |
| 500 | No-reverse PD | 1.41% | 0.64% | 38.28 | 1.22 | 9.25% | 14.44% | 1.48 | 0.00 | 0.00 | 13.32 | 5.45 |
| 500 | 原生 HiCache/Mooncake | 2.69% | 3.88% | 13.11 | 2.25 | 7.03% | 8.20% | 2.38 | 0.00 | 0.00 | 6.14 | 8.30 |

KV和Mamba为各引擎设备池的不可驱逐使用率均值；queue、inflight、running、prealloc和transfer均为每引擎均值。
Colocated共享同一套KV池和调度队列，故只在D侧列一次，P侧共享项记为“—”。

## 结果解释约束

- c256 Colocated 的首次尝试 `...c256-20261009-r1` 在209/500时因baseline将线程栈
  提高到约64MiB、并发Triton JIT触发 `pthread_create/posix_spawn EAGAIN` 而作废；`r2`
  在启动期发现仍加载环境内旧包，尚未发正式请求即清理作废。两份原始产物均保留供审计；
  修复仅显式绑定baseline源码并限制CPU/JIT线程及线程栈，正式数据从全新 `r3` 重跑。
- c256 No-reverse PD 的首次尝试 `...c256-20261009-r1` 完成490/500个episode回调后作废：
  workload进程继承了非交互Slurm步骤的1024文件描述符软上限，导致10个并发SWE工作树
  报 `Errno 24: Too many open files`。启动器现会在创建服务与workload前把软上限提高到
  `min(65535, hard limit)`；正式重跑使用 `r2`，不会混入该轮数据。
- c256 HiCache/Mooncake 的首次启动 `...c256-20261009-r1` 未发送正式请求即作废：
  H100 TP1在`mem_fraction_static=0.8`下的设备Attention KV池约30.14 GiB、Mamba池
  约15.06 GiB，原32/16 GB host tier因十进制GB与GiB差异不满足stock HiCache的
  `host > device`不变量。`r2` 改用40/24 GB每rank后通过启动，但在60/500时因8个
  worker的数百线程、glibc arena和线程栈共同推高`Committed_AS`，触发Triton/LLVM
  `pthread_create failed`，同样作废。`r3` 限制glibc arena为2并把线程栈降到2MiB后，
  仍因首次动态Triton编译被推迟到正式负载而在54/500触发`ptxas posix_spawn EAGAIN`。
  `r4` 将本地tier收敛到仍合法的34/18 GB并恢复PD fake-bootstrap warmup，但源码把动态
  token数`T`误标为Triton `constexpr`，导致新T仍不断JIT，两个Prefill在111/500退出。
  `r5` 将`T`改为运行时维度并移除未使用的`NB constexpr`，运行至273/500未发生服务
  故障；但审计发现Triton仍会按整数对齐生成两个变体，故主动清理、不纳入正式结果。
  `r6` 进一步采用FLA相邻kernel已有的`do_not_specialize=["T"]`，使所有动态T只
  复用一个cubin，8个GPU服务稳定收齐500个回调；但SWE容器/验证器在Host高水位下有
  88题遭遇临时fork/thread `EAGAIN`，因此整轮作废。`r7` 对未创建子进程的Popen EAGAIN
  做有界重试、关闭Git threaded preload，并由控制器强制拒绝任何environment error。
  `r7` 随后暴露出更早的Apptainer首次物化阶段也会在Host commit高水位触发fakeroot
  fork EAGAIN，运行至322个回调后主动作废。`r8` 将同类无副作用的启动失败纳入有界重试，
  并把仍高于设备池的host tier由34/18收敛到33/17 GB每rank，释放16 GB commit余量。
  `r8` 将启动错误降至47题，但审计发现Apptainer还有若干等价的间接报错文本，且Git
  命令偶尔返回0却被profile fork警告污染stdout，故在335回调时作废。`r9` 补全这些
  无副作用启动重试，并只对baseline/patch-capture的幂等Git探针重试；控制器健康检查
  也改走现有SSH，避免高commit时创建新Slurm step失败。前八轮产物均保留供审计。
- c394 HiCache/Mooncake 的 `r1` 在232/500后出现1个sandbox broker启动失败：c394下
  同时活跃的sandbox更多，原12次、约20秒的无副作用启动重试窗口仍不足。该轮立即
  作废；`r2` 将broker尚未ready前的重试扩展为最多40次、约70秒，其他实验参数不变。
- c500 当前新方法的 `r1` 收到498个回调后，两个普通chat请求因router先等待未使用的
  Prefill HTTP响应、而不是已完成的Decode响应，滞留至3600秒读超时；重试同一不可变
  generation后触发ownership守恒检查，Prefill-0按设计退出。该轮作废，不纳入正式
  数据。`r2` 对非-logprob请求以Decode响应作为用户结果，在finally中取消并回收悬挂
  的Prefill HTTP task；18项router/controller回归测试通过，模型、采样、并发和PD配置
  均未改变。
- 这是500题有限数据集测评，不是300秒预热+1200秒持续补请求的闭环稳态测试。
- T500容易被少数verifier或长任务主导，因此方法比较优先看T450及T50–T450主体段。
- 每组详细Forward占比、P/D吞吐、KV/Host水位和新方法ownership守恒在原始metrics基础上
  复算；任何基础设施异常必须在表中注明，不能静默并入正式结论。
