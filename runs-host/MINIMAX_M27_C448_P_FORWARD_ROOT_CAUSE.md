# MiniMax c448 P Forward 根因诊断与修复方案

## 结论

**2026-10-17 用户澄清验收目标：** 当前 P 算力有余量，优化重点不是缩短
T300/T450，而是减少 P 为相同任务实际消耗的 Forward GPU 时间，使之明显低于
HiCache。Forward 占比与累计 GPU 时间必须同时报告，不能靠延长等待降低占比。
T300 未改善不再作为单独停止/否决依据；以下 r2/r3 的停止基于此前标准。
新旧方法的 workload/shape 差异仍须单独列出，不把全部 event 时间差当成计算节省。

已修复计算门禁的等待判据和 completion 漏唤醒，通过 227 项回归及独立审计。
但两组双节点 c448 工程测试均未改善 T300（关闭门禁 1,056.59 s、仅修复判据
995.05 s，原组 980.61 s），已按用户允许的终止条件停止，不能宣称性能问题解决。

目前不能判定为“恢复 KV 在硬件上不划算”或“必须增加节点才能解耦”。短 trace
显示 rank 0 host enqueue 迟到；终点后 GIL 栈采样进一步发现控制消息 deepcopy/
JSON 处理热点，但二者均不足以分摊正式窗口总耗时。r4 已减少复制/广播编码，
r5 又优化页链哈希，并实测 Host 准备阶段：截至 4,049 次恢复，描述符准备 p95
仅约 2.89 ms，未出现合成碎片测试的百毫秒开销。下一步更值得验证的是多阶段
重复携带完整控制 payload，而非重写 Host 描述符数据面；详见末尾。

本报告对应 2026-10-17 MiniMax-M2.7 TP8/EP8、双节点 1P:1D、SWE500、c448。
P 与 verifier 仍在同一节点，不使用第三节点。下文诊断描述修复前代码；用户批准后
已实施最小门禁修复，实测进度见末尾，尚未修改描述符与缓冲池路径。

## 已确定的现象

| T50–T450 指标 | Agentic-PD | HiCache/Mooncake |
|---|---:|---:|
| 新增 Prefill token | 4,916,754 | 18,564,751 |
| P Forward event 时间/GPU | 721.49 s | 672.83 s |
| P Forward batch 数 | 1,848 | 3,850 |
| 平均 new token/batch | 2,662 | 4,822 |
| 平均 cached token/batch | 45,477 | 16,819 |
| 平均 event 时间/batch | 390.4 ms | 174.8 ms |
| T450，从各自 workload 起点计 | 1,428.60 s | 1,483.18 s |

`1848 / 3850 × 390.4 / 174.8 ≈ 1.072` 是可靠的总量核账：Agentic 的 batch 数减半，
但每批 event 时间增加到约 2.23 倍。它不是这个增加来自何处的因果分解。

之前给出的“236 ms/batch、436.7 s/GPU 特有开销”是跨实现微基准插值后的残差，
不是实测组件时间；“53% 来自传输”也只是混杂的区间相关性。两项因果归属均撤回。
原始测量保留在 [微基准报告](MINIMAX_M27_C448_P_FORWARD_MICROBENCH.md)。

## 明确的代码问题

### 计算门禁把排队也当作物理传输

正式 launch manifest 开启 `serialize_p2d_direct_with_forward=true`。
[scheduler.run_batch](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/managers/scheduler.py:10172)
在每个 P rank 进入 Forward 前调用 `wait_p2d_direct_physical_idle`，默认最多等 1 秒。
注释意图是只等已经开始的 P→D Direct DMA。

但 [wait_physical_idle](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_group_transfer.py:548)
实际只检查 `local.transfer is not None and local.completion is None`。
[_start](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_group_transfer.py:683)
在入 I/O 队列之前就设置 `local.transfer`；真正 posted 的通知通过后续
`_submitted` / `submit_ack_sent` 表示。谓词没有使用后者。

因此 START 后仍在排队、等 source-ready fence 或建链的操作也会挡住计算。
在 node1800 用真实方法、构造 rank-local 状态、`timeout=0` 做了最小 CPU 复现：

| 状态 | 按函数声明应视为 physical idle | 实际返回 |
|---|---|---|
| transfer 已建立，physical-start ACK=false，completion=None | 是 | **否** |
| physical-start ACK=true，completion=None | 否 | 否 |
| completion 已到达 | 是 | 是 |

这是语义不匹配的复现，不是完整运行重放。正式 P 日志没有触发 1 秒 serialization
deadline 的警告，但这不能证明没有等待，甚至不能排除下面这种睡满 timeout 后
返回 idle 的情况。

另一个相关问题是 `_start` 的 no-I/O 和提交异常分支：设置 `local.completion`
之后没有通知 `_changed` condition。已有 waiter 可能继续睡到 timeout，最后检查
到 idle 又返回 True，因此不打印 deadline warning。用同一真实等待方法做 CPU
复现，20 ms 后设置 completion、timeout=200 ms：不通知时等待 **200.09 ms**，
通知时等待 **20.20 ms**，两者都返回 True。这证明漏通知可制造无警告等待；没有
证据说明正式成功 Direct 路径频繁命中这些特殊分支，不把它计入本轮开销比例。

### rank 独立等待可能把时间转移进 Forward 计时

该门禁不是 TP 组统一决策。若 rank A 在门禁前等待较久，rank B 已开始 Forward，
B 可以在下一次 TP collective 等 A。A 的门禁等待位于 timer 外，B 的 collective
等待却位于 timer 内。因此不能因为门禁代码在 timer 外，就认定它不可能抬高
平均 Forward event 时间。

示例仅用于解释机制：若一个 rank 迟到 100 ms，另外七个 rank 都在已计时的
collective 中等待，则最多可表现为约 700 GPU·ms 的等待、约 87.5 ms/GPU 的
平均 event 增量。实际值取决于何时遇到 collective、CUDA 提交和可重叠计算；
**这不是本轮测出的 87.5 ms**。正式日志缺少各 rank 的门禁起止与 collective trace。

建议先关闭这项可选的计算门禁做受控 A/B，保留所有 snapshot 的真实 ready/fence/
ownership 检查。若保留开关，必须先修正“已排队”和“已物理提交”的判据，并补齐
completion 状态改变后的 condition 通知。仅在谓词
里增加一个状态位也不构成严格的传输互斥：检查返回后，新 DMA 仍可能开始；不要把
这个性能开关当成内存安全 fence，更不要为此引入跨 rank 的全局计算屏障。

## 可以优化但尚未证实为主因的准备开销

### Host 恢复描述符的 Python 工作量随页碎片放大

生产 [_host_target_payload](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_default_physical_provider.py:884)
把 GPU token-index vector 拷到 CPU，逐 token 构造 Python tuple。
[RemoteHostRankWorker.load](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_remote_host_worker.py:346)
随后执行 `mha_read_spans`，合并连续 token runs，并按全部 K/V 层展开。
[prepare_read](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_remote_host.py:604)
再逐 span 构造 local/remote/destination lists、排序检查重叠、建立 NIXL descriptors。

MiniMax 有 62 层，每 rank 1 个 KV head；29,504 tokens 共 461 个 64-token pages。
若全部连续，仅需 124 个 spans；若每页不相邻，会展开为 `124 × 461 = 57,164`
个 spans。这些 Python 工作在同一 scheduler 进程的后台线程中，仍竞争 GIL。
给线程多分 CPU core 不能消除这个进程内的 Python 执行竞争。

调用现有 `mha_read_spans` 和 `prepare_read` 的 CPU 隔离测试如下。每项 10 次，舍弃
首轮，分别取两阶段中位数；运行在登录节点，NIXL agent 用无操作 stub 替代，页布局
为合成。没有计入实际 NIXL API、GPU→CPU index mirror 和真实 verifier 负载。

| snapshot tokens | 页布局 | spans | 构造 spans | prepare 的 Python 部分 |
|---:|---|---:|---:|---:|
| 29,504 | 全连续 | 124 | 6.00 ms | 0.12 ms |
| 29,504 | 每页不相邻 | 57,164 | 29.37 ms | 56.10 ms |
| 49,088 | 全连续 | 124 | 9.07 ms | 0.12 ms |
| 49,088 | 每页不相邻 | 95,108 | 60.72 ms | 105.80 ms |
| 131,072 | 每页不相邻 | 253,952 | 156.04 ms | 305.98 ms |

这证明当前实现对碎片敏感，**没有证明正式运行具有上述碎片程度或耗时**。
正式 restore trace 未记录 span_count，因而不能把 85 ms 直接加到每个生产 batch。

最小方案是从 lease 的不可变 CPU page/run plan 建立描述符，避免逐 token 镜像和
多轮 Python 遍历；已有 `mha_page_read_spans` 可作为接口起点，但单纯改用它仍保留
按层展开与 `prepare_read` Python 循环。后者应使用连续数组或释放 GIL 的 native
实现，同时保留范围、完整性和目标不重叠检查。只缓存同一有效 lease 内的不可变
元数据，不跨已释放 lease 复用地址。先不引入新进程、CUDA IPC 或新的 ownership 层。

### D2H 每个 snapshot 重建未必使用的缓冲

[CudaSourceHostCopyBackend.copy](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/disaggregation/agentic_source_host.py:411)
每次新建 CUDA stream、`LayerFirstD2HStaging(2048)` 和 pinned bounce。
MiniMax TP8 下后二者各约 62 MiB/rank。registered indexed DMA 成功时，数据直接
进入最终 Host extent，不使用这两个 fallback buffer，CPU commit 也是 no-op。

node1800 的独立 GPU 小测试：先预热 5 次，再测 10 次构造及同步，墙钟中位数
**0.228 ms**；另用 profiler 采样 3 次，见 6 次 HtoD pointer copy、6 次
`cudaStreamSynchronize`、3 次 arange kernel。它证明存在无效初始化和 stream 同步，
不证明 device-wide sync；也不能用 0.228 ms 解释每批约 266 ms 的模型残差。
profiler 中额外的 device synchronize 来自测试收尾，不归因到生产构造函数。

最小方案：每条 lane 持久复用 stream；仅在提交前确认 batch DMA 不支持时懒建
fallback staging，Host 未注册时再懒建 pinned bounce。fallback buffer 按 lane
独占并复用，不能让多个 snapshot 共用正在 DMA 的 buffer。

方向必须区分：D→P offload 的构造发生在 D，不直接解释 P Forward；P→D offload
发生在 P，才可能直接干扰本报告的 P 指标。不能把双向所有 CPU/DMA 开销都记到 P。

## 不能当作代码 bug 消除的成本

1. 复用 prefix 省掉旧 token 的投影、MLP 和 KV 生成，但新 query 仍须读取 prefix
   做 Attention。忽略常数时，该部分与 `Σ(q_i × c_i + q_i × (q_i+1)/2)` 相关，
   不能只看新增 token 总数，也不能仅凭 batch 的两个总长度还原逐请求计算量。
2. MiniMax MoE 的真实 expert routing、rank imbalance、collective 和每 batch 固定
   成本仍存在。随机 token、平均分配长度的重放没有保持这些因素。
3. DMA 与 Forward 最终共享 GPU 内存访问资源；跨节点还共享 NIC/PCIe 路径。
   独立 stream 或后台线程不等于物理带宽隔离，stream 优先级也不是带宽配额。

这些成本不意味着“必须串行”。是否重叠有利要看恢复与计算的总关键路径，而非单个
Forward event 是否变短。把全部 DMA 等完再算，可能使 Forward 数字更好却让 T450
更差。MiniMax 本轮是 full attention，不应拿 Mamba checkpoint 复制解释差额。

## 现有微基准能证明和不能证明的内容

本地 p50 shape 下，完整重算约 800.85 ms，resident Forward 约 85.60 ms，本地
Host 恢复约 42.77 ms。这支持该合成场景中复用比重算便宜；不代表真实跨网恢复的
全部 latency，也不构成正式运行的端到端逐请求对照。

正式 V2 走 `RemoteHostLoadExecutor → RemoteHostRankWorker.load → NIXL READ`，
直接从源 DRAM 写入目标 VRAM。微基准却是 legacy 本地 1024-token H2D pipeline，
没包含 P→D D2H、双向 Direct、真实页碎片、group controller 与 verifier。HiCache
微基准行也只采用了其 shape，并未执行原生 HiCache 恢复实现。

[DeviceTimer](/home/shangy/dualpd/sglang-src/pd/python/sglang/srt/utils/device_timer.py:15)
测两次 CUDA event 之间的时间，不是计算 kernel 活跃时间之和。CPU 发射 gap、
collective 等待和带宽争用可能都在里面；Forward 之前的 ready/queue 等待通常在
外面，但可能通过 rank 偏斜间接进入其他 rank 的 event 区间。

补测 `ForwardBatch.init_new` 的 isolated envelope 未见大幅变慢，故没有证据将
metadata 构造列为主因。这个 envelope 仍不是完整生产 worker wrapper，且没有
真实并发负载，不能由此排除全部生产 CPU/采样开销。

## 实施与验收顺序

| 顺序 | 动作 | 判断依据 |
|---|---|---|
| 1 | 原配置短测中记录逐 rank 的 pre-forward gate 起止、Forward NVTX、CPU launch gap、collective、真实 NIXL post/fence；采样真实逐请求 q/c、span_count 和 fallback 计数 | 首先区分 kernel 变慢、TP 等待、CPU 发射停顿；不能只依赖 event 总时间 |
| 2 | 保持 snapshot 安全 fence 不变，关闭可选 P→D 计算门禁做同配置 A/B；补测试修正其未-post判据 | 若 rank 偏斜和 collective 等待下降且总时间改善，证明这项门禁有负收益；若 DMA 干扰增加则如实记录 |
| 3 | 按测出的占比优化 page/run descriptors 与 lazy per-lane buffer 复用，每项单独验证 | CPU 构造时间、span_count、GIL/launch gap 与 Forward event 是否一起下降；不凭源码推断收益 |
| 4 | 仅当 kernel/collective 与真实 DMA overlap 显著关联时，再单独降低 D2H 或 Direct 物理并发 | 保持独立队列和有界推进，不加全局 Forward 等 DMA 的新屏障；同时检查 HBM retention 和恢复等待是否恶化 |
| 5 | 工程短测通过后重跑双节点 c448，主指标 T450，附 T300 和 T450 内 decode 吞吐 | 用户要求 T300 无改善即可停止；不靠降低复用率、改变题集或挪走 verifier 改善数字 |

匹配 shape 的原因分解必须使用同一份真实 token、逐请求长度及页映射，测 resident、
真实 NIXL 并行、P→D D2H 并行和生产双向组合。不同方法的生产分布用于加权，而不是
拿不同分布直接做因果减法。恢复带宽用实际完成字节数除以真实 post→fence 时间，
单独报告描述符准备、admission、全 rank commit 和 ready→Forward 排队。

应优先用现有 profiler 进行有限窗口采样；profile 会扰动运行，因而 profile 样本
用于归因，关闭 profile 的运行用于 T450 对照。AGENTS 要求的 300 秒预热加 1200 秒
测量是正式性能验收门槛；更短的诊断只标为工程测试，不包装成正式吞吐结果。

## 不改变的所有权约束

方案仅改变传输准备和可选性能门禁，不改变 snapshot 主所有者和 TP 组决策。

- P→D Direct 的源 P HBM 仍须等真实 DMA 完成及既有交付协议才能释放。
- P→D / D→P D2H 的源 HBM 仍须等完整 snapshot durable；不等目标将来有空。
- Host→GPU 必须已有完整 workset，全部 rank 恢复成功并提交后才释放源 Host。
- lane、描述符数组、indices、注册窗口和 buffer 保留至 fence/drain 完成。
- 已发生可能的物理提交后，失败必须 drain/quarantine，不能直接切 fallback 重传。
- 工具未返回不提前恢复；terminal snapshot 不复活；P→D 不驱逐；TP 所有 rank
  保持单一逻辑决定。取消、容量不足、部分 rank 失败和退出继续 fail-closed。

实际实现前后须通过生命周期、取消/未-post/部分-post、容量和 TP 测试，独立审计
GO 后才能 GPU 测试。结束时检查 Host/pending/tombstone 清零与 source-release 守恒。
本报告没有宣称这些尚未实现的优化已经通过性能或状态机验收。

## 修复前判断与待验证方案

**可修的工程问题确实存在，最优先的是计算门禁的语义与 TP 偏斜；但它尚未被证明
解释全部 Forward 差额。没有证据支持现在就增加硬件或宣布无法优化。** 先在现有
两节点内减少不必要等待和 Python 传输准备，再用真实 NIXL trace 判定剩余的共享
资源成本。如果修正后确实主要剩长前缀 Attention、MoE/collective 或物理带宽竞争，
则不能承诺消除它们，也不能承诺新增 token 减少 3.8 倍就得到 3.8 倍 Forward 加速。

原始生产数据目录：

- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-agentic-swe500-c448-20261017-r1`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c448-20261017-r1`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-c448-p-microbench`

源码 checkout HEAD 为 `4b353f9142db666dd95664fddc1346b012022197`，存在本地修改，
不能把此 SHA 当作生产全部代码的唯一指纹。关键 source_host/provider/host_staging
文件的当前修改时间早于本轮生产运行；该事实有助于核对，但不替代运行时源码快照。

## 修复与实际测试记录

2026-10-17 首次实现仅包含三处生产变更：`wait_physical_idle` 使用
`submit_ack_sent` 判定已提交；no-I/O 与提交异常两个分支在同一 Condition 锁内
设置 completion 并通知等待者。没有修改 source-ready、物理 fence、commit/release
或 TP 决策。五项新增回归覆盖 queued、posted、scope、完成唤醒、no-I/O、提交失败
以及源内存不能提前释放。

回归结果：149 项传输/生命周期/内存/故障测试通过，另有 78 项 group protocol、
physical provider、scheduler shell 测试通过，合计 **227 passed**。仅有依赖弃用
警告。独立审计结论 **GO**。八项不变量复核：唯一所有者、两方向 durable 后释放、
Direct 交付释放、独立推进、TP 原子性、父 KV 复用及测试审计门禁均未改变。

测试 run：`minimax-m27-tp8ep8-agentic-c448-20261017-gateoff-r2`。
首次较长名称触发启动器 64 字符校验，未启动模型；缩短名称后重试。
使用原 job `26049949`、node1800 与 node2701。规范化 run-id 派生路径与控制 token
之后，r1/r2 manifest 唯一配置差异是 `serialize_p2d_direct_with_forward: true → false`。
这是关闭额外计算门禁的单因素测试，不将它与后续描述符优化合并归因。

| 对照 | T300 | T450 | 状态 |
|---|---:|---:|---|
| Agentic c448 r1 | 980.61 s | 1,428.60 s | 历史对照 |
| HiCache c448 r1 | 1,012.13 s | 1,483.18 s | 历史对照，代码未改 |
| Agentic c448 gateoff r2 | 1,056.59 s | — | T300 较 r1 慢 7.75%；已停止，不外推 T450 |
| Agentic c448 gatefix r3 | 995.05 s | — | 较 r1 慢 1.47%；已停止，不外推 T450 |

### r2/r3 工程测试：均未取得性能改善

两组使用同一 T50–T300 区间定义、按 counter 插值计算；GPU 时间除以 8，
不与前面的 T50–T450 表混用。

| 指标 | 原 Agentic r1 | gateoff r2 | gatefix r3 |
|---|---:|---:|---:|
| T50–T300 墙钟时间 | 685.02 s | 768.73 s | 717.01 s |
| P Forward / GPU | 498.32 s | 652.11 s | 536.48 s |
| P Forward 占比 | 72.75% | 84.83% | 74.82% |
| 实际新增 Prefill tokens | 3.767 M | 4.130 M | 3.892 M |
| P 新增 Prefill 吞吐 | 5,498.87 token/s | 5,372.38 token/s | 5,427.46 token/s |
| Prefill cache token 比例 | 92.88% | 93.07% | 92.92% |
| D Forward 占比 | 97.77% | 98.07% | 98.22% |
| Decode tokens | 1.156 M | 1.304 M | 1.190 M |
| Decode 吞吐 | 1,688.08 token/s | 1,695.73 token/s | 1,659.34 token/s |
| D 平均 running requests | 34.57 | 34.98 | 33.33 |
| P / D 平均 KV 使用率 | 27.79% / 38.97% | 30.12% / 39.63% | 27.04% / 36.79% |

**不能把 T300 慢 7.75% 全部归为 DMA 干扰。** 相同采样配置不保证不同调度下
生成轨迹完全相同；r2 该窗口 decode 工作量多约 12.7%，decode token/s 基本持平。
但 P 每新增 token 所耗 Forward 时间也没有改善，完全去掉门禁不能作为解决方案，
不改默认值。r2 到 T300 无失败和 environment error。

T300 之后采集了全部 8 rank 的短窗口 CPU/GPU trace；采样以后的耗时不可用于
无 profile 的性能对比。随后按组件顺序停止本 run，确认两节点 GPU 进程及本 run
进程均为 0，并卸载本 run 的 158 个 FUSE 挂载。提前停止不是 500 题自然 drain，
不能据此宣称 Host lifecycle 自然清零验收通过。

r3：`minimax-m27-tp8ep8-agentic-c448-20261017-gatefix-r3`。规范化 run-id/token
后 manifest 与 r1 相同，保留 `serialize_p2d_direct_with_forward=true`，只使用
上述通过 227 项回归和独立审计的等待修复。本轮暂无性能改善结论。

r3 同样未改善 T300，按用户允许的终止条件停止。T300 前没有开启 profiler；
到第 300 题无失败/environment error。停止前进度为 322/500，同样无上述错误，
余下 178 题被主动终止，不能当作 500 题完成结果。确认两节点 GPU 及本 run 进程
均为 0，卸载本 run 178 个 FUSE 挂载；原 Slurm 分配保留，没有占用第三节点。

两次工程测试均短于 300 秒预热 + 1200 秒测量的正式门槛，不替换主结果表。
保留语义正确性修复，**不将完全关闭门禁设为默认，也不声称效率问题已经解决**。

已归档两组的 P/D、Router、group control、workload 日志到各 run 的
`collected-logs/`。Router `application_final` HTTP 记录分别为 r2：372 次 200、
r3：344 次 200，均无 503；这是接口调用数，包含预热/终止期间事件，不能直接当作
题目完成数。两组未发现 `D2P Host pressure` 记录。`d2p_host_evictions` 计数也
包含 application-final 清理，不把它全部解释成压力驱逐。

| 最后一次 policy 日志（非一致性 fence 快照） | r2 | r3 |
|---|---|---|
| D→P Host entries / GiB | 15 / 53.433 | 88 / 270.039 |
| D→P Host 状态 | durable 7、restore submitted 8 | durable 17、restore submitted 71 |
| P→D 状态 | restore submitted 41、durable 32、capacity wait 7、direct submitted 8 | restore submitted 13、direct submitted 10 |

这些为主动停止时尚未 drain 的 snapshot，不能宣布全部正常清零，也不能只凭这张
表判为泄漏。两组没有完成逐 snapshot `queued/durable/source_release` 终态守恒
验收；确认的是全进程退出后的 GPU 资源释放。不得把本次工程测试当成生命周期
完整长测通过。

### r2 短 trace：先追查 rank 0 发射迟到，不把 collective 等待算成计算

原始 trace 位于 r2 的 `profile-after-t300/`，8 rank 文件均包含 7 个完整 graph、
625 个 NCCL all-reduce、约 14.8k kernels。独立审计复核了以下观察：

| 短样本指标 | P rank 0 | P ranks 1–7 |
|---|---:|---:|
| trace 墙钟跨度 | 4.409 s | 4.290–4.534 s |
| collective kernels 时长之和 | 208.7 ms | 3,198–3,488 ms |
| Attention kernels 时长之和 | 90.8 ms | 90.8–92.0 ms |
| MoE 分类 kernels 时长之和 | 162.7 ms | 162.2–164.4 ms |
| profiler Command Buffer Full overhead | 0 | 约 574–948 ms |

按照相同顺序对齐 CPU all-reduce annotation，rank 0 在 625 次中的 578 次最晚
到达；7 个 graph 入口有 6 个晚于其他 rank 的中位数。该样本支持 host enqueue
迟到/TP 等待，而不支持 rank 0 的 Attention/MoE kernel 本身显著变慢。

**这仍不是正式窗口的归因比例。** Profiler 有明显且不对称的扰动；文件没有共同
batch-id，顺序对齐不是严格分布式因果证明。NIXL RDMA 不一定显示为 CUDA kernel
或 memcpy，因此 kernel union 之外不能叫“HBM 空闲”。CPU annotation 迟到也
不能直接归因为 GIL：OS 调度、TP0 控制工作、CUDA API 阻塞均可能造成它。
下一步使用短窗 Python/GIL 采样与线程 CPU 时间核对，且放在性能终点之后。

### r3 终点后 Python/GIL 采样：重复控制消息处理是实测候选

T300 之后对 P rank 0/1 执行 `py-spy record --gil --nonblocking --rate 50`，
没有改变模型代码或测量终点。rank 1 得到 235 个有效样本、103 次采样错误；
rank 0 得到 353 个样本、192 次错误，采样在 worker 停止时结束，两个采样窗口
不能当作严格等长。文件为 r3 下 `gil-after-t300-rank{0,1}.raw`。

| 有效 GIL 栈样本分布 | P rank 0 | P rank 1 |
|---|---:|---:|
| MainThread | 101 | 86 |
| group control receive 线程 | 76 | 84 |
| 固定 coordinator/control-owner 线程 | 91 | 不承担 coordinator |
| `_frozen_payload` 的 deepcopy 调用栈 | 30 | 68 |

表内最后一行包含在对应线程样本内，不能相加。采样丢失、非阻塞读取及 GIL 过滤
使这些数值不能转换为墙钟时间比例，更不能说“控制开销占了 47% Forward”。
它们确实把代码候选进一步缩小到控制消息复制/编解码及 coordinator 工作：
`GroupCommand.from_dict → __post_init__ → _frozen_payload` 会再次递归 deepcopy
刚从 JSON 解码出的 payload，随后再 JSON 序列化验证；多轮 handoff 还携带原 plan。
rank 0 另外承担 ACK/状态推进。下一步优先测试减少重复 payload 复制/序列化，
保持消息不可变、校验、幂等和 TP fence，不先增加 DMA 并发或把 verifier 移走。
本次尚未修改这部分代码，不能预报其收益。

补充核对：本次 HiCache c448 manifest 使用同一个 `pd` Python 环境，源码 checkout
为 `pd_baseline_b1ef6`（记录的 SGLang 版本 0.5.10.post1），不是先前 Qwen 实验的
v0.5.21 环境。MiniMax 模型与 fused-MoE 两个源文件 hash 与 Agentic 相同；这不等于
全部 scheduler/communication 路径相同，但排除了这两个文件的版本差异。

## 第二轮：控制消息复制与广播编码优化（r4）

用户要求继续修复后，2026-10-17 实施两项纯 CPU 控制路径优化：

1. `_copy_control_payload` 对 exact builtin dict/list 及不可变标量使用快速复制，
   保留 memo、共享子对象、循环、source keepalive；tuple、subclass 和其他类型
   回退原 `copy.deepcopy`。消息仍深复制隔离，并执行原严格 JSON 校验。
   用于 protocol payload 以及 plan/command/DMA result 的原有复制点。
2. relay `_broadcast` 原来对 TP8 P/D 的 16 个接收 rank 各做一次 JSON 编码。
   现在编码一次，逐 socket 锁下发送同一份不可变 bytes。保留 endpoint/membership
   校验与异常传播；不缓存跨消息数据、不改 wire format、顺序或 ACK 协议。

没有更改数据面、路由、source-ready/DMA fence、workset、Host 容量、工具阈值或
任何 ownership phase。广播本来就不是原子网络多播：部分 socket 失败时，已收到
命令的 rank 仍按原协议持有/回滚资源，coordinator 不能在缺 ACK 时提交。本改动
不把 partial send 当作成功，也不增加新的重试/超时或提前释放。

八项门禁复核：①唯一 owner 不变；②P→D Direct 仍等全部 rank 交付；③④两向
Host durable 后的源释放不变；⑤不添加计算等待，队列独立性不变；⑥固定
coordinator/TP phase 不变；⑦不改变 parent 内容、页计划或重算规则；⑧最终代码
14 个生命周期/故障/容量/协议/TP 测试文件 **270 passed**，仅依赖弃用警告。
新增覆盖 alias/cycle/tuple/subclass/keepalive、非法 JSON、随机嵌套数据、16-rank
单次编码、缺 rank、未知 endpoint、部分 socket 失败。独立审计已核对代码哈希、
测试结果和八项不变量，结论 **GO**；允许启动 r4 单因素 GPU 测试。

CPU 微测在 node1800 使用实际 helper；以下为合成 page-hash/Host-descriptor
结构，110 次取去前 10 次后的中位数。NIXL metadata 大小是假定值，并非线上采样。
不把这些毫秒数乘生产 batch 数推算整轮节省。

| context tokens / 每 rank metadata | 消息 bytes | 复制＋校验：原 → 新 | 16-rank 广播编码：原 → 新 |
|---|---:|---:|---:|
| 32,768 / 0 | 35,145 | 0.1538 → 0.0913 ms | 1.0666 → 0.0654 ms |
| 32,768 / 8 KiB | 100,681 | 0.2424 → 0.1818 ms | 2.5335 → 0.1541 ms |
| 131,072 / 8 KiB | 203,594 | 0.5984 → 0.3858 ms | 5.9311 → 0.3209 ms |

r4：`minimax-m27-tp8ep8-agentic-c448-20261017-ctrlcopy-r4`。仍用 job 26049949，
node1800 P+verifier / node2701 D，MiniMax TP8/EP8、SWE500、c448。规范化 run-id
和 token 后 manifest 与 r3 完全一致；保留修正后的 P→D 等待门禁。
2026-10-17 15:13 UTC 开始启动；最初计划以 T300 筛选，随后按用户澄清改为
优先比较累计 P Forward GPU 时间，结果如下。

关键文件 SHA256：protocol `2fb2f5e135f1d61f5e8dec4cf03cacd4aabb874a1d51c6229fa69bf2659b00bb`，
group_transfer `92e76739996b0d2fe0342e18a8c769b3ac5439d9242706fee7ef7d267c7f9f72`。

### r4 已完成 T300：按澄清后的 GPU 时间目标评价

T50=314.04 s、T100=458.86 s、T200=727.10 s、T300=992.86 s。
原组 T300=980.61 s，r3=995.05 s。T300 之前没有任务失败或 environment error。
用户在此时澄清主目标为减少 P Forward 资源消耗，不再仅因 T300 无提升停止。

| 各自 T50–T300 窗口 | 原 Agentic r1 | 门禁修复 r3 | 控制复制优化 r4 | HiCache c448 |
|---|---:|---:|---:|---:|
| 窗口墙钟 s | 685.02 | 717.01 | 678.82 | 571.49 |
| 实际 Prefill M tokens | 3.7668 | 3.8916 | 3.7690 | 15.9631 |
| cached M tokens | 49.1137 | 51.0719 | 50.3659 | 32.5265 |
| P Forward s/GPU | 498.32 | 536.48 | 463.92 | 469.00 |
| P Forward 占比 | 72.75% | 74.82% | 68.34% | 82.07% |
| 新 token / P Forward GPU 秒 | 7,559 | 7,254 | 8,124 | 34,036 |
| 计数器复用率 | 92.88% | 92.92% | 93.04% | 67.08% |
| D Forward 占比 | 97.77% | 98.22% | 98.42% | 99.72% |
| Decode M tokens | 1.1564 | 1.1898 | 1.2180 | 1.0027 |
| Decode token/s | 1,688.08 | 1,659.34 | 1,794.36 | 1,754.58 |
| D running requests，时间平均 | 34.57 | 33.33 | 39.23 | 36.98 |
| P / D KV 池占用 | 27.79% / 38.97% | 27.04% / 36.79% | 22.77% / 44.23% | 11.77% / 82.09% |

r4 对比 r1 的 Prefill 数近乎相同，P Forward s/GPU 降约 6.9%，是值得保留的
观察；但没有重复实验/同 token、同 page-plan 重放，不能把差值全部归因于补丁。
更重要的是，r4 仍以 HiCache 约 23.6% 的 Prefill token 消耗其约 98.9% 的
P Forward 时间。因此不能宣称用户关注的效率问题已解决。占比低本身不充分，
还须看墙钟分母、累计 GPU 时间、cached Attention 工作量和 collective/launch 等待。

T300 原始数据已复制到 r4 的 `engineering-t300-checkpoint/`。T300 后进行了
py-spy 诊断：rank0 输出 393 有效样本、222 错误；rank1 未按 20 秒正常结束，
已单独终止该 profiler，未终止 worker。T300 前未开 profiler；后续窗口受诊断
影响，不能当成无采样的正式 T450 对照。当前仍是工程测试，尚无自然排空/ownership
守恒验收结论；未替换正式主结果表。

### T0–T300 补充口径

| 指标 | 原 Agentic r1 | 修改后 r4 | HiCache |
|---|---:|---:|---:|
| T300 s | 980.61 | 992.86 | 1012.13 |
| P Forward s/GPU | 729.12 | 699.60 | 764.84 |
| P Forward 占比 | 74.35% | 70.46% | 75.57% |
| 实际 Prefill M tokens | 5.45 | 5.60 | 18.92 |

Agentic 启动约13秒后才首次输出 Forward 计数，按零起点累计；约5秒/GPU的
起点边界不确定性，不影响趋势。r4 相比原 Agentic 少约4.0%、相比 HiCache 少约
8.5%的 Forward 时间；之前 T50–T300 的“几乎相同”不能代表整个 T0–T300。

r4 后续继续到 T450=1397.71 s，停止前493/500、terminal FAILED=0；后段有
`Server disconnected` 自动重试，且 T300 后开过 py-spy，故 T450 不作干净性能
对照。停止前最近 policy 的双向 phases/pending 均空、D2P Host entries/GiB=0；
这不是逐snapshot source-release守恒审计。工程停止后两节点GPU进程/本run进程
连续两次均为0，6个本run FUSE卸载；日志已归档到`collected-logs/`，没有删除数据。

## 第三轮：页链哈希优化与真实 Host 准备阶段诊断（r5）

r4 终点后 rank0 的393个有效 GIL 栈中，20个包含`token_page_chain_hashes`：
10个来自下一轮request_control_intent，10个来自Prefill完成source_snapshot。
这些均在controller线程，和Forward共享进程GIL。采样有222次错误，另有1个
无关栈含乱码；不能把20/393当成生产墙钟占比，更不能据此分摊GPU总时间。

明确代码冗余：每个token都`int.to_bytes`再`SHA256.update`。现改成每页
`tuple(map(int, page)) + struct.pack('<Nq')`，一次update；累计哈希字节流、
完整页边界及signed-int64范围保持一致，不缓存跨request状态、不改变复用证明。
合成CPU测试（node1800，同解释器）32768/49088/131072 tokens分别约
3.46→1.41 / 5.08→2.15 / 13.69→5.72 ms。这是函数隔离收益，不是端到端收益。

更大但尚未实证的候选是Host描述符的逐token/逐span Python遍历：源代码及合成
碎片测试证明其复杂度，尚缺真实span_count与耗时。已核对已安装NIXL支持Nx3
NumPy descriptor输入，若线上确实受此限制，可用连续数组替代多轮Python tuple
展开；在得到真实数据前不改该数据面。

r5增加显式`profile_host_prepare=true`诊断（默认false），在Host worker记录
span_build、transport_setup、prepare_read、post、read_wait墙钟时间与span数量。
不加CUDA同步、不修改状态机；成功poll+retire后记录，普通logger异常不影响receipt。
caller的GPU→CPU index mirror与group commit/source release不在内；read_wait不是
物理DMA时间。同步logging会有扰动，所以r5只作诊断，不能用于正式性能验收。

八项不变量复核：①唯一owner不变；②P2D Direct释放不变；③④两向Host源释放
仍依赖原durable/fence；⑤无新增计算/I/O屏障（诊断有日志开销）；⑥TP全rank
ACK和commit不变；⑦页链逐字节一致；⑧16个SGLang生命周期/故障/TP/证明测试
文件及launcher测试共337 passed，独立审计通过后才可启动GPU。

r5配置规范化run/token后，相对r4仅新增`profile_host_prepare=true`，同节点
node1800 P+verifier / node2701 D、SWE500 c448、MiniMax TP8/EP8及全部容量不变。
run：`minimax-m27-tp8ep8-agentic-c448-20261017-hostprofile-r5`。

### r5 实测诊断快照（2026-10-17 16:10 UTC，尚未结束）

独立审计已给出诊断运行 GO，337 项 CPU 测试通过后启动；workload 起点约
16:01:47 UTC。16:09:35 时完成 119/500，T100=402.34 s，terminal FAILED 与
environment_error 均为 0。此时不能给出 T0–T300 收益，更不能将它作为正式性能
结果；同步诊断日志有扰动。截至随后检查 application-final 152 次均 HTTP 200、
503 为 0，但出现 12 次 `Prefill backend cleanup failed: ServerDisconnectedError`，
不能宣称全程零异常。未自然排空，尚无本轮完整 ownership 守恒验收。

下表来自 P worker 的真实 D→P Host 恢复日志，每个 snapshot 有 8 个 rank 样本。
阶段时间是 host wall time，不是 CUDA kernel/DMA 活跃时间。

| 指标 | 全部 4,049 次恢复 / 32,392 rank 样本 | 其中 ≥16,384 tokens 的 104 次 / 832 rank 样本 |
|---|---:|---:|
| snapshot tokens，均值 / 最大 | 5,977 / 29,760 | 19,318 / 29,760 |
| spans，p50 / p95 / 最大 | 124 / 124 / 620 | 124 / 248 / 496 |
| span 构造，均值 | 0.929 ms | 2.805 ms |
| prepare_read，均值 | 0.446 ms | 0.549 ms |
| 准备总时间，p50 / p95 | 1.226 / 2.890 ms | 3.050 / 5.159 ms |
| post 后等待，p50 / p95 | 53.301 / 152.877 ms | 190.254 / 306.395 ms |

准备总时间包含 transport 初始化；最大值约 1.21 s，主要对应首次建立 transport，
不能当作每次恢复的成本。GPU→CPU index mirror 在调用者中、不在表内；group
commit/source release 在返回后，也不在表内。post 后等待含 progress/poll/调度，
不是纯 DMA 时间。当前还没有 ≥32K 单 snapshot 样本，不能外推到全部后期长上下文。

**排除/收窄：** 真实页布局基本连续。合成测试中 57K–254K spans 的严重碎片
并未出现，因此目前不实施 NumPy descriptor 重写；其潜在收益不足以解释每批
数百毫秒的差额。此前“可能存在大量 Python 描述符开销”仅为候选，现已降优先级。

### 网络与控制消息：两个不同的限制

只读采样 node1800 的 IB port counters，间隔 30.107 s，按 IB 四字节单位换算：
`mlx5_1` 接收 16.814 GB/s、发送 17.974 GB/s；其链路速率 200 Gb/s（理论单向
25 GB/s）。其余 ACTIVE 的 400 Gb/s ports 在本窗口几乎没有流量。这是节点端口
总流量，混合 Direct/Host 等路径，不是某条恢复路径的独占带宽，也不是持续饱和证明。
虽然进程环境没有显式 UCX_NET_DEVICES，实际流量仍集中在该端口；不能仅根据
manifest 的 ib_device 推断 NIXL 使用哪块网卡。其他端口是否具有可用跨节点路径
还未验证，不能直接声称换网卡可获得 2 倍收益。

同窗口 `ss -tinp` 中本 run group-control TCP 连接的字节增量：P rank0 接收
3.631 MB/s、发送 3.310 MB/s；其余 P rank 各接收约 2.172 MB/s、发送约
0.057 MB/s。它量化了 rank0 额外控制工作的数据规模，**不等于 GIL 时间占比**。
`/proc` 线程 CPU 采样没有可靠的 Python 线程名映射，不将其归因到具体函数。

网络读等待通常位于 Forward 之前；它可以限制供给、拖长工具返回后的恢复，但
不能把这段时间直接加到 P Forward。对用户关心的 Forward 资源浪费，更相关的
仍是 rank0 host enqueue 偏斜/collective 等待及控制工作，而非仅让网络更快。

### 下一项有针对性的优化设计（后续已实施，见 r6）

源码中 `AgenticGroupTransferOrchestrator` 的成功收尾把同一
`state.handoff_payload = {rank_results, transfer}` 依次附加到 RELEASE、HANDOFF、
ACTIVATE、ISSUE_ACTIVATION_TICKET、FINALIZE 五个广播阶段。r4 只消除了每个
广播重复编码 16 次，尚未消除这五个阶段重复传输，以及各 worker 重复解析、
复制和 JSON 校验完整 payload。下一步应优先减少**重复元数据**，而非取消 ACK/fence。

建议在既有 attempt 生命周期内保留一次完整 transfer 与一次完整 receipts，后续
阶段只携带 generation/attempt/lease/command-sequence 引用；接收方严格验证对应
attempt 后使用已经验证的元数据。保持所有原有阶段和全 rank ACK，不压缩状态机，
不改变物理传输、不让计算等 I/O、不提前恢复工具尚未返回的 KV。

实施门禁：必须验证内部 payload 的不可变性和所有消费者，再补未知引用、跨 attempt
串用、重复/乱序、取消、断连和 FINALIZE 回收测试。数据只活在已有 attempt 中，
不得跨 lease 缓存地址；失败仍按既有物理 fence 收敛。CPU 测试和独立 GO 前不启用
GPU。验收看相同进度 T0–T300/T450 的累计 P Forward、真实 workload/shape、
rank 间等待与控制成本同时下降，不能以降低占比或提高 T100 单独宣布成功。

如果元数据去重后仍无明显下降，下一步是低扰动、同 batch-id 的逐 rank launch/
collective 诊断；只有确认模型 kernel 或真实 DMA overlap 的贡献后才调整物理 I/O
并发。当前证据不足以认定必须物理拆分资源，更不支持增加第三节点。

## r6：收尾控制元数据引用优化

用户批准后实施 `compact_handoff_payload`，显式默认关闭；launcher 为所有 worker
统一设置 `SGLANG_AGENTIC_COMPACT_HANDOFF_PAYLOAD=0/1`。本轮只优化成功交接的
四个尾部命令，不改 PREPARE/START、取消、DMA、ACK 或 ownership 状态机。

- RELEASE 仍完整发送；rank 在既有 `_LocalAttempt` 内保存一份独立深拷贝，
  必须在 handler callback 前建立，不让 callback 修改引用基准。
- HANDOFF、ACTIVATE、ISSUE_ACTIVATION_TICKET、FINALIZE 只发
  `{handoff_payload_ref: RELEASE.command_seq}`。接收端严格核对 key/run/generation、
  attempt、lease、owners、endpoint_groups、固定阶段顺序和精确序号 R+1…R+4。
  拒绝 bool/string 引用、额外 payload 字段、未知/过期引用、取消后的引用。
- 还原仍调用原 GroupCommand 构造器进行深拷贝和 JSON 校验，保留当前 kind/seq，
  因而 ACK 不会误用 RELEASE 序号。dispatcher 将还原命令交给 after_command，
  避免丢失下一轮 target_generation。未消除全部接收端复制成本。
- 引用数据随原 FINALIZE/ABORT_FINALIZE 的 attempt 清理，不形成跨请求/跨 lease
  地址缓存。版本/开关不一致 fail closed，不猜测缺失 payload。

八项不变量：①不新增物理 owner；②③④仍沿原 Direct/Host fence 和全rank ACK
释放；⑤仅改变后台控制元数据表示、不增加 Forward/I/O 等待；⑥所有 TP rank
执行同一原阶段/ACK；⑦parent proof 与 KV 字节不变；⑧17个相关SGLang文件及
launcher测试 **361 passed**，包括引用隔离/乱序/取消、Host及时释放、GPU source
scheduler ticket、真实TCP TP1/2/8 next-generation传递。独立审计 GO 前不启动GPU。
登录节点首次 pytest 因没有 libcuda 无法导入 sgl_kernel；迁到已有 node1800 跑
同一套 CPU 生命周期逻辑测试后通过，未修改环境或安装依赖来绕过。

### CPU 编解码隔离测试

node1800、pd Python；合成 512/2048 页哈希、8 rank 各 8 KiB metadata；模拟五阶段
sender→relay→16 receivers，使用实际 GroupCommand/JSON frame 编解码，compact
接收端仍做独立复制。预热1次、重复12次取中位数。不含真实 socket、ACK、DMA、
调度，也非生产 payload 的完整重放；不能按这个百分比预估 Forward 收益。

| 页哈希数 | 模式 | sender ms | relay ms | 16 receivers 合计 ms | 合计 CPU ms | 控制 wire bytes |
|---|---|---:|---:|---:|---:|---:|
| 512 | 原完整 payload | 1.746 | 3.330 | 24.693 | 29.785 | 8,529,606 |
| 512 | compact 引用 | 0.435 | 0.787 | 20.472 | 21.694 | 1,723,894 |
| 2048 | 原完整 payload | 3.771 | 7.198 | 54.677 | 65.625 | 17,277,126 |
| 2048 | compact 引用 | 0.835 | 1.555 | 45.145 | 47.522 | 3,473,398 |

五阶段控制流量减少约80%、所模拟CPU总时间减少约27%；接收端仍有显著剩余成本。
这些 CPU 时间是顺序模拟的工作量，不是实际并发16进程的墙钟延迟。

### r5 对照收尾与 r6 配置

r5 T300=944.246 s，T0–T300 P Forward=678.794 s/GPU（71.887%），新增Prefill
5.3091 M、cached 59.3193 M、Decode 1.5347 M。r4 对应699.596 s/GPU，但 r5
实际 token 工作量也较少且有同步诊断日志，不能将差值全部归于页链哈希优化。
T300原数据保存在`engineering-t300-checkpoint/`。随后停止本run，双节点GPU进程与
本run进程连续两次均为0，卸载197个本run FUSE挂载，未删除模型/数据/结果。
最后policy仍有P2D restore/store/direct=36/2/7，D2P restore/direct/wait_child=90/2/9，
Host90entries/251.224GiB；工程中断不是自然排空，不能报ownership清零验收。

r6：`minimax-m27-tp8ep8-agentic-c448-20261017-compactmeta-r6`，仍job26049949、
node1800 P+Router+verifier / node2701 D。规范化run-id/token后与r5仅
`compact_handoff_payload=true`不同；保留 `profile_host_prepare=true` 使诊断口径
一致。这是工程A/B，不替换正式结果。以相同任务进度的累计 P Forward 和实际
token/cached工作量判断，不单看利用率或T100。

独立代码/状态机审计已给出 **GO**；审计者另在现有node1800运行18个相关文件，
347 passed（CUDA_VISIBLE_DEVICES关闭，5项平台/依赖warning）。同质launcher下
身份、复制隔离、ACK、next-generation、取消/部分故障、终态清理及八项不变量
均通过。16:26 UTC 左右开始启动r6，GPU实测结论待补。

16:31:46 UTC，r6 workload 已运行106.3秒、完成7/500，terminal FAILED=0、
environment_error=0；P/Router/workload当前未见Traceback、CUDA错误、引用错误或
HTTP503，application-final已观察到15次HTTP200。已从两节点全部16个scheduler
进程的实际环境核对compact/profile均为1。此为早期功能验证，非性能结论；
T0–T300累计Forward和完整ownership排空尚待完成。

r6代码SHA256：group_transfer
`d323f893d667b662378e9a02dd874b337dbeb1a952e6996707c94eed3657fc53`；runtime
`55d369fc83499b9338f75560af2efe08415e1b1e1e2b979d6c878fa38a42537c`；launcher
`d74b6827b8d447ea76efb851c3680024d08ef269fd0f8151e10cb19bce8324e0`。
