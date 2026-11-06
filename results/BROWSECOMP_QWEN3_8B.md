# BrowseComp + Qwen3-8B

固定 source-order n680 循环，temperature=0，8卡、TP=1，300秒业务预热 + 1200秒正式测量。
显存按物理GPU对齐：普通卡0.80，GPU7（搜索服务）0.60；PD为4P:4D，P=0/2/4/6，D=1/3/5/7。

## H100重测（2026-10-06，node1701）

本节单独记录H100结果，不覆盖下文A100历史数据。统一使用固定source-order n680、seed2026、
temperature0、top_p1、top_k-1、context40960、轨迹预算36864，以及300秒业务预热＋1200秒正式测量。
不重测vLLM和PPD，共12组；顺序为新方法、Colocated、No-reverse、原生Mooncake。

新方法统一使用D→P Host总量512 GiB（128 GiB/P）和P→D Host总量256 GiB（64 GiB/P），
共768 GiB注册窗口；2条D2H lane和2条H2D lane，4096-token chunk，且全部Host注册完成后才开始业务预热。
原生Mooncake保留其原生共享segment/HiCache机制，不人为拆成方向容量；其实测物理分配另行记录。

| 方法 | 并发 | 状态 | Decode | Agent/s | Prefill token/s | Parent KV复用 | 完成/失败 |
|---|---:|---|---:|---:|---:|---:|---:|
| 当前新方法（Slow拥堵反馈重算） | 384 | 完成 | 10,719.7 token/s | 5.3249 | 74,372.2 | 98.51% | 6,390 / 0 |
| 当前新方法（Slow拥堵反馈重算） | 512 | 完成 | 10,634.1 token/s | 5.2117 | 78,625.9 | 95.34% | 6,254 / 0 |
| 当前新方法（Slow拥堵反馈重算） | 576 | 完成 | 10,778.5 token/s | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |
| Colocated baseline | 384 | 完成 | 10,622.4 token/s | 5.3133 | 106,248.2 | 76.26% | 6,376 / 0 |
| Colocated baseline | 512 | 完成 | 9,993.3 token/s | 4.9800 | 128,014.6 | 58.30% | 5,976 / 0 |
| Colocated baseline | 576 | 完成 | 9,329.5 token/s | 4.6842 | 137,779.3 | 47.08% | 5,621 / 0 |
| No-reverse PD | 384 | 完成 | 4,877.0 token/s | 2.4100 | 112,429.5 | 0.00% | 2,892 / 0 |
| No-reverse PD | 512 | 完成 | 4,899.7 token/s | 2.4467 | 111,359.0 | 0.00% | 2,936 / 0 |
| No-reverse PD | 576 | 完成 | 5,297.8 token/s | 2.6199 | 113,299.2 | 0.00% | 3,144 / 0 |
| 原生Mooncake | 384 | 完成 | 5,417.8 token/s | 2.6742 | 105,482.5 | 17.11% | 3,209 / 0 |
| 原生Mooncake | 512 | 完成 | 5,331.9 token/s | 2.6350 | 106,450.0 | 14.64% | 3,162 / 0 |
| 原生Mooncake | 576 | 完成 | 5,358.5 token/s | 2.6508 | 107,662.5 | 13.28% | 3,181 / 0 |

H100新方法c384有效轮为301.236秒预热＋1200.022秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡2,679.917 token/s，Decode/Agent 2,010.79 tokens，实际Prefill/Agent 13,950.71 tokens，
无请求失败、HTTP 5xx、服务崩溃或cgroup内存事件。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c384-w300-m1200-20261006-r2`。
此前r1仅进入约2分钟业务预热，因1000 GiB作业cgroup触发`memory.events max`而主动废弃；
重提1800 GiB作业后r2全程`memory.events max=0`，r1不计入结果。

H100新方法c512为300.879秒预热＋1200.000秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡2,658.514 token/s，Decode/Agent 2,034.82 tokens，实际Prefill/Agent 15,045.01 tokens。
窗口尾段出现两次独立的前端单次连接重试，均未进入attempt 2，8个P/D健康检查均为200，最终请求失败0；
有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c512-w300-m1200-20261006-r1`。

H100新方法c576为300.618秒预热＋1200.001秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡2,694.624 token/s，Decode/Agent 2,040.80 tokens，实际Prefill/Agent 15,013.42 tokens。
预热和测量期各出现一次独立的前端单次连接重试，均未进入attempt 2，最终请求失败0；有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`。

H100 Colocated c384为300.907秒预热＋1200.000秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,327.804 token/s（8卡），Decode/Agent 1,995.56 tokens，实际Prefill/Agent 19,960.10 tokens，
请求失败0；有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c384-w300-m1200-20261006-r1`。

H100 Colocated c512为301.267秒预热＋1200.000秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,249.164 token/s（8卡），Decode/Agent 2,002.62 tokens，实际Prefill/Agent 25,653.61 tokens，
请求失败0；有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c512-w300-m1200-20261006-r1`。

H100 Colocated c576为300.597秒预热＋1200.002秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,166.192 token/s（8卡），Decode/Agent 1,986.50 tokens，实际Prefill/Agent 29,336.80 tokens，
请求失败0；有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c576-w300-m1200-20261006-r1`。

H100 No-reverse c384为301.738秒预热＋1200.001秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,219.258 token/s（4张Decode卡），Decode/Agent 2,019.33 tokens，
实际Prefill/Agent 46,551.19 tokens，Parent KV复用为0；完成2,892个Agent，请求失败0，
且无HTTP错误、服务崩溃或cgroup内存事件。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c384-w300-m1200-20261006-r1`。

H100 No-reverse c512为300.867秒预热＋1200.001秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,224.913 token/s（4张Decode卡），Decode/Agent 1,996.88 tokens，
实际Prefill/Agent 45,384.94 tokens，Parent KV复用为0；完成2,936个Agent，请求失败0，
且无HTTP错误、服务崩溃或cgroup内存事件。运行期间更新launcher的Mooncake可选参数后，旧Bash进程
在analyzer完成后读取文件尾部时报语法错，因此wrapper最终退出码为2；边界、原始数据、摘要和图均已完整落盘，
不影响本轮测量有效性。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c512-w300-m1200-20261006-r1`。

H100 No-reverse c576为301.850秒预热＋1200.030秒正式测量，边界状态为`steady_churn`；
正式窗口Decode每卡1,324.461 token/s（4张Decode卡），Decode/Agent 2,017.14 tokens，
实际Prefill/Agent 43,138.40 tokens，Parent KV复用为0；完成3,144个Agent，请求失败0，
且无HTTP错误、服务崩溃或cgroup内存事件。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c576-w300-m1200-20261006-r1`。

H100原生Mooncake c384为300.677秒预热＋1200.007秒正式测量，边界状态为`steady_churn`；
显式按物理GPU绑定本地NUMA，并使用1200秒scheduler watchdog；其余保留原生机制：256 GB共享segment、
P HiCache 128 GiB/实例、D HiCache 56 GiB/实例和P write-through。正式窗口Decode每卡
1,354.452 token/s（4张Decode卡），Decode/Agent 2,022.38 tokens，实际Prefill/Agent 39,374.82 tokens，
Parent KV复用17.11%；完成3,209个Agent，请求失败0。Mooncake窗口末驻留212.9/256.0 GB，
45/45次eviction成功、allocation failure为0；cgroup内存峰值量级约1003 GiB且无内存事件。
有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c384-w300-m1200-20261006-r2`。
此前r1因Slurm shell未包含baseline环境的CUDA runtime库路径，`mooncake_master`在模型启动前退出，
未产生业务数据；r2补入环境内`libcudart.so.12`路径后通过全部健康检查，r1不计入结果。

H100原生Mooncake c512为301.641秒预热＋1200.001秒正式测量，边界状态为`steady_churn`；
保持与c384相同的原生256 GB共享segment、P/D HiCache容量、本地NUMA绑定和1200秒watchdog。
正式窗口Decode每卡1,332.964 token/s（4张Decode卡），Decode/Agent 2,022.58 tokens，
实际Prefill/Agent 40,380.70 tokens，Parent KV复用14.64%；完成3,162个Agent，请求失败0。
Mooncake窗口末驻留215.27/256.0 GB，46/46次eviction成功、allocation failure为0；
cgroup内存峰值量级约1003 GiB且无内存事件。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c512-w300-m1200-20261006-r1`。

H100原生Mooncake c576为300.018秒预热＋1200.020秒正式测量，边界状态为`steady_churn`；
保持与c384/c512相同的原生256 GB共享segment、P/D HiCache容量、本地NUMA绑定和1200秒watchdog。
正式窗口Decode每卡1,339.637 token/s（4张Decode卡），Decode/Agent 2,017.97 tokens，
实际Prefill/Agent 40,544.41 tokens，Parent KV复用13.28%；完成3,181个Agent，请求失败0。
Mooncake窗口末驻留196.66/256.0 GB，46/46次eviction成功、allocation failure为0；
cgroup内存峰值量级约1002 GiB且无内存事件。有效结果目录为
`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c576-w300-m1200-20261006-r1`。


## 当前完整新方案

2026-09-09起采用 **快慢路径＋全局Slow恢复拥堵反馈重算**，对应c512实测 **5,148.4 token/s**。
c384/c576均已用当前完整方法重新完成。

- 工具超过1秒未返回：Slow；工具1秒内返回：尝试Direct，建链deadline为1秒。
- 快工具Direct失败：恢复队列不拥堵时转Slow；拥堵时显式完整重算。
- 全局Q只计工具已返回、Host durable、尚未被H2D worker接手的唯一parent generation；跨P不重复，等待工具不计入。
- 每秒采样，连续两次Q≥32进入拥堵，Q≤8退出；信号超过3秒陈旧或无效时保守Slow。
- 只决定新发生的Direct失败出口，不取消已在Host中的请求，不抢占在途KV；DMA fence、workset和TP规则不变。
- Q是等待worker接手的代理指标，不是实际DMA等待队列；32/8是已测参数，不宣称最优。

新方案复现须设置 `SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=true`、`SLOW_CONGESTION_HIGH=32`、
`SLOW_CONGESTION_LOW=8`（后两项完整前缀同为`SGLANG_AGENTIC_KV_`）。
原生HiCache/Mooncake关闭；D→P Host=128 GiB/P、P→D Host=32 GiB/P、H2D=2 lanes/P，
D接收目标1.0，P→D grace=0.5秒；全部Host预注册完成后才开始业务预热。
No-reverse和Colocated基线保留已经完成的配置对齐结果。原生Mooncake旧c384/c512
未显式绑定本地NUMA，已从正式对比口径撤下，需按c576相同的本地NUMA配置重跑。

## 实验矩阵

c384 四种方法（含原生 HiCache + Mooncake）的时间序列可视化见[组合图](figures/browsecomp-qwen3-8b-c384/c384_dynamics_combined.png)；
分项图包括[父 KV 命中率](figures/browsecomp-qwen3-8b-c384/kv_hit_rate.png)、
[时间加权Decode活跃请求](figures/browsecomp-qwen3-8b-c384/decode_batch_size.png)、
[Decode发生时的条件batch](figures/browsecomp-qwen3-8b-c384/decode_batch_when_active.png)和
[384 个请求的阶段分布](figures/browsecomp-qwen3-8b-c384/request_phase_counts.png)；
三种方法的逐阶段直接对比见[阶段曲线](figures/browsecomp-qwen3-8b-c384/request_stage_comparison.png)。
指标定义和复现方式见[图表说明](figures/browsecomp-qwen3-8b-c384/README.md)。

2026-09-10额外c576消融：已回退统一NUMA池到SGLang `921fbd46ab6d`，
只关闭P→D Host、保留late binding和D→P全部路径，正式300+1200秒为
**4965.7 token/s**，较旧独立Host完整方法4822.5提高2.97%。本轮2874 Agent完成、0失败，
D Forward 99.43%、P Forward 93.87%、D running/卡47.54、D KV/卡69.0%。
不据此替换完整方案默认设置；单轮差异及3个预算终止Host副本残留见
[本轮完整报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1/RESULTS.md)。

| 方法 | 配置 | 状态 | Decode |
|---|---|---|---:|
| Colocated baseline | 8卡，c384 | 完成 | 4,834 token/s |
| Colocated baseline | 8卡，c512 | 完成 | 4,434 token/s |
| Colocated baseline | 8卡，c576 | 完成 | 4,349 token/s |
| No-reverse PD | 4P:4D，c384 | 完成 | 1,982 token/s |
| No-reverse PD | 4P:4D，c512 | 完成 | 1,984 token/s |
| No-reverse PD | 4P:4D，c576 | 完成 | 2,153 token/s |
| 原生 Mooncake | 4P:4D，c384 | 完成（显式本地NUMA） | 2,049.5 token/s |
| 原生 Mooncake | 4P:4D，c512 | 完成（显式本地NUMA） | 1,966.7 token/s |
| 原生 Mooncake | 4P:4D，c576 | 完成（显式本地NUMA，见下） | 2,041 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c384 | 完成 | 4,737.0 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c512 | 完成 | 5,148.4 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c576 | 完成 | 4,822.5 token/s |
| 原生 vLLM PD（NIXL） | 4P:4D，c512 | 完成（2026-09-17） | 2,136.17 token/s |
| PPD-Dynamic（作者方法，vLLM/NCCL） | 4P:4D，c512 | 完成（2026-09-17） | 1,346.08 token/s |

2026-09-10补齐原生Mooncake c576：使用未修改的`pd_baseline`引擎，300.54秒业务预热＋1200.00秒正式测量。
本轮在独立启动脚本中显式按物理GPU绑定本地NUMA，并把scheduler watchdog从300秒延长到1200秒；
旧c384/c512启动日志为全部TP1 worker自动绑定NUMA0。因此缓存容量、模型和数据参数对齐，
但CPU/NUMA放置并不完全一致，不能把c576相对旧行的吞吐差异仅归因为并发变化。
未修改共享启动脚本、harness、`pd_baseline`源码或另一个agent的自定义引擎。

2026-09-17补齐原生vLLM PD c512：301.492秒业务预热＋1200.001秒正式测量。
使用`vllm`环境（vLLM 0.11.0、PyTorch 2.8.0+cu128、Transformers 4.57.1、NIXL 0.5.1），
按本机driver 570固定版本；PPD使用独立`vllm_ppd`环境，结果见下；vLLM与PPD的c384/c576尚未测试。
模型与数据配置对齐本表：Qwen3-8B BF16，context40960、轨迹预算36864、单轮2048、最多100轮，
temperature0/top_p1/top_k-1/seed2026，原BrowseComp source-order n680与搜索harness；
P/D物理卡和0.80/0.60显存比例同上。复现入口为`CONCURRENCY=512 bash examples/pd/scripts/vllm/run_pd.sh`。

原生vLLM采用round-robin P/D、prefix caching、chunked prefill，block_size16、
max_num_batched_tokens8192、max_num_seqs256；每轮P准备一次单token请求后传原生KV grant给D，
丢弃P准备token并保留原始prompt，最终输出来自D；缺失grant报错，适配层不做本地Prefill回退。
GPU0–6实际KV容量各339,648 tokens，GPU7为224,224 tokens。
SGLang page64、自定义路由与vLLM设置不同，显存比例相同不等于实际KV容量相同；
GPU7原有ComfyUI进程约588 MiB全程保留。本轮未做参数搜索，仅为一次正式结果，不能代表vLLM最佳性能。
已有新方法c512 Decode是本轮的2.410倍，差异包含引擎与路由因素，不能全部归因于单个KV机制。
原repo embedding pickle不完整，本轮只读使用同一搜索数据集revision
`9f600f47c5ee9a6251ec5521eb279d8dc5df2966`的完整HF cache（100195文档），未修改原pickle。

2026-09-17补齐 **PPD-Dynamic c512**：300.448秒业务预热＋1200.001秒正式测量。
使用[作者代码](https://github.com/freelulul/vllm-ppd) commit `28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367`；
`vllm_ppd`为vLLM 0.13.0 release / PyTorch 2.9.0+cu128加作者NCCL connector及明确记录的兼容修复。
作者vendored基线`b95d054de59c948e55ea18397d328ad79b8df630`并非0.13.0 release同一提交，
因此本行标为作者方法适配，不宣称逐提交原样复现；也不是与上行vLLM 0.11.0的同版本算法消融。
未改作者仓库、SGLang新方法或共享BrowseComp harness；独立入口为
`CONCURRENCY=512 bash examples/pd/scripts/vllm_ppd/run.sh`，方法、修复与验收说明见
[PPD适配说明](../scripts/vllm_ppd/README.md)。模型、数据、工具、采样、预算、物理卡与显存比例均沿用本表。

PPD保留作者原始决策器及固定P/D hash亲和：首轮PD，后续按TTFT/TPOT等权评分选择PD或D本地，
input<512走作者短输入旁路；使用原工作负载分类、累计QPS、最近QPS表项和长context外推。
用作者生成器/测量程序在本机完成40个静态校准点，得到20个PD/PPD配对profile（large_mid_bal与large_huge_paste，各10个QPS）；没有缺失profile回退。
这组实测决策在正式窗口将全部2066个后续请求送往D本地，796个首轮请求走PD；并非强制静态PPD，但本轮没有实际覆盖后续轮次的PD/PPD切换效果。
其中553次直接查表、1454次长context外推、59次短输入旁路。

修复限于异步NCCL物理完成/暂存生命周期、原生抢占元数据兼容和P端图执行。
P使用PIECEWISE以保证全前缀命中时仍执行逐层KV发送；D保留FULL_AND_PIECEWISE。
此前正式r1在重复样本上未收到KV，已废弃，不计入本结果。
图模式变更后实际KV容量与校准完全一致（普通卡339,648、GPU7为224,224 tokens）；
保留原校准值及[模式兼容说明](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/profiles/producer-graph-compatibility.json)，不宣称图模式性能等价。
12项逐token预检覆盖四组P/D、跨轮复用、长prompt分块，以及P命中/D冷缓存的真实36层传输。

本次PPD Decode比原生vLLM PD低 **36.99%**；已有新方法是它的 **3.825倍**。
D端高KV占用、频繁抢占和低Parent复用与cache thrashing及重复Prefill相符，
但本次没有单独计量物理Prefill重算量，也不能把跨引擎版本差异全部归因于PPD策略。
短测的高吞吐未延续到正式长测；本行是指定c512配置的一次测量，不代表PPD调参后的最佳表现。

## 正式窗口吞吐与数据特性

| 方法 | 并发 | Agent/s | Prefill token/s | Decode/Agent tokens | 实际Prefill/Agent tokens | Parent KV复用 |
|---|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 384 | 2.392 | 38,149 | 2,014 | 15,893 | 91.74% |
| Colocated baseline | 512 | 2.181 | 46,523 | 2,031 | 21,308 | 74.11% |
| Colocated baseline | 576 | 2.160 | 47,552 | 2,008 | 21,957 | 74.31% |
| No-reverse PD | 384 | 0.978 | 42,976 | 2,022 | 43,846 | 0.00% |
| No-reverse PD | 512 | 0.966 | 42,803 | 2,051 | 44,251 | 0.00% |
| No-reverse PD | 576 | 1.019 | 42,680 | 2,106 | 41,753 | 0.00% |
| 原生 Mooncake | 384 | 0.975 | 37,164.2 | 2,102 | 38,115 | 17.30% |
| 原生 Mooncake | 512 | 0.952 | 38,489.3 | 2,061 | 40,336 | 14.73% |
| 原生 Mooncake（本地NUMA） | 576 | 0.993 | 37,850 | 2,052 | 38,054 | 15.13% |
| 当前新方法（Slow拥堵反馈重算） | 384 | 2.333 | 34,486 | 2,029 | 14,769 | 97.48% |
| 当前新方法（Slow拥堵反馈重算） | 512 | 2.387 | 35,195 | 2,153 | 14,719 | 96.30% |
| 当前新方法（Slow拥堵反馈重算） | 576 | 2.334 | 35,472 | 2,062 | 15,165 | 95.86% |
| 原生 vLLM PD（NIXL） | 512 | 1.0275 | —（未取得实际计算量） | 2,079 | —（未取得实际计算量） | 5.99% |
| PPD-Dynamic（vLLM/NCCL） | 512 | 0.6633 | —（未取得实际计算量） | 2,029 | —（未取得实际计算量） | 5.97% |

Parent复用按完成轨迹的完整page-aligned父前缀统计；新方法显式重算会降低该比例。Prefill/Agent与Decode/Agent为窗口总计算量除以完成Agent数。

vLLM Decode为D原生生成计数减首token计数：2139.778−3.608=2136.170 token/s，
P准备token不计入；窗口内完整计数器采样覆盖1196.520秒（560组/角色），无缺失引擎或计数重置。
逻辑prompt为44,439.903 token/s，包含缓存命中，不能填入实际Prefill列；
summary内嵌旧分析器的`actual_prefill_tokens`仅为完成轨迹的prompt−cache，
`page_aligned_extra_prefill_from_parent_kv_loss`也是推算值，都不是物理计算计数。
Parent复用只取P本地缓存命中，不计P→D远程传输：1,983,168/33,086,352=5.994%（page16）。

vLLM正式窗口结束1233个Agent：505正常finish、728按原配置截断（447单轮length、281轨迹budget），
请求及后端失败均0；其中841个多轮Agent，共4098次模型调用、2334次search、885次open_page。
全部已完成轨迹符合source-order n680循环；Agent/s含长度/预算结束，不等于正确率。

PPD Decode按相同口径为1348.456−2.376=1346.080 token/s；原生计数采样覆盖1197.838秒，
每角色564组采样。窗口结束796个Agent：324 finish、288单轮length、184轨迹budget，0失败；
534个多轮Agent，共2563次模型调用、1468次search。Agent/s包含按配置截断，不是解题正确率。
Parent复用为窗口内结束轨迹的完整历史：1,210,688/20,274,704=5.971%（page16），
包括这些轨迹在窗口前发生的轮次；PD取P本地命中、D本地路径取D本地命中，不把远程KV传输当缓存命中。
Decode/Agent列是窗口token流速除以Agent完成速率，不是已结束轨迹的实际平均输出长度。
逻辑prompt 28,989.672 token/s以及vLLM的prompt−cached衍生计数均不能当作包含抢占重算的物理Prefill量。

## 稳态资源

资源为正式窗口各引擎采样均值；Forward为每物理卡平均。Colocated的KV池由P/D共享。

| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated c384 | 51.1% | — | — | — | 48.8% | 67.6% | 46.3 | 0.8 | — | — |
| Colocated c512 | 60.4% | — | — | — | 39.6% | 84.1% | 59.9 | 3.1 | — | — |
| Colocated c576 | 62.0% | — | — | — | 37.9% | 84.6% | 66.1 | 4.8 | — | — |
| No-reverse PD c384 | 99.41% | 7.96% | 17.62 | 0.72 | 98.28% | 92.27% | 7.61 | 0.00 | 68.27 | 20.02 |
| No-reverse PD c512 | 99.20% | 7.92% | 16.84 | 0.72 | 98.59% | 92.01% | 8.63 | 0.00 | 100.05 | 19.25 |
| No-reverse PD c576 | 98.42% | 7.84% | 17.37 | 0.74 | 99.15% | 91.99% | 8.72 | 0.00 | 115.40 | 19.80 |
| 原生 Mooncake c384 | 99.20% | 9.72% | 16.88 | 1.04 | 98.87% | 92.09% | 8.34 | 0.00 | 67.82 | 19.73 |
| 原生 Mooncake c512 | 99.47% | 9.37% | 16.64 | 0.94 | 98.36% | 92.18% | 8.13 | 0.00 | 100.31 | 19.45 |
| 原生 Mooncake c576（本地NUMA） | 97.26% | 9.15% | 17.52 | 1.02 | 99.03% | 91.94% | 8.21 | 0.00 | 115.33 | 20.33 |
| 新方法 c384 | 90.04% | 67.08% | 25.56 | 8.70 | 98.46% | 79.32% | 50.33 | 0.00 | 1.43 | 0.24 |
| 新方法 c512 | 88.99% | 55.01% | 55.76 | 9.37 | 99.07% | 74.91% | 52.48 | 0.00 | 1.65 | 0.20 |
| 新方法 c576 | 90.57% | 57.27% | 75.10 | 8.41 | 99.11% | 76.87% | 49.67 | 0.00 | 1.57 | 0.21 |
| 原生 vLLM PD c512（NIXL） | — | 15.72% | 99.19 | 1.00 | — | 38.59% | 24.00 | 2.35 | — | — |
| PPD-Dynamic c512（vLLM/NCCL） | — | 0.00% | 0.00 | 0.00 | — | 93.47% | 68.90 | 58.39 | — | — |

原生Mooncake c576正式窗口结束1192个Agent：486正常完成、706按原配置截断（422单轮长度、284预算），
无`aborted`、无记录的请求/搜索后端错误；共3834次模型调用，平均3.216轮。
396条一轮轨迹中381条为单轮长度终止、15条正常结束，不是此前搜索失败导致的全部一轮退出。
Agent/s沿用本表“无请求错误的结束Agent（含截断）/正式时间”口径，不等于正确率或完整解题率。
Mooncake正式窗口平均占用80.38%、峰值84.90%；全程累计成功驱逐46次、分配失败0。

vLLM资源行按正式窗口时间加权、每引擎平均；P inflight映射原生running，P/D queue映射原生waiting，
与SGLang的细分调度状态不完全等价。Forward、D prealloc/transfer未取得同口径指标，以“—”表示而非0。
正式窗口P/D四引擎合计抢占分别4/15次。主要队列位于P；吞吐及占用随窗口变化明显，
满足正式时长门槛不等于通过统计平稳性检验，见[时间序列](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/vllm_pd_timeseries.png)。

vLLM全程四个D成功NIXL transfer分别1538/1538/1537/1537，共6150；
代理守恒为submitted6569=completed6057+cancelled512，failed0、active0。
窗口末取消419个P阶段请求、93个D阶段请求；后者为已收到但未确认D完成的grant，
前者可能尚未收到grant。这些协议计数不等同于物理KV泄漏/释放证明。
原生running/waiting队列排空后全部自有服务退出，仅原有ComfyUI GPU进程保留。
本轮10项CPU/取消/故障测试及独立代码、结果审计通过；原harness、数据和配置运行前后哈希一致。

PPD资源行沿用每引擎时间加权口径，正式P/D抢占分别0/1306次。P只执行首轮短Prefill，
且循环样本可命中P缓存；两秒级采样的P running/waiting/KV使用均为0，不意味着没有P计算或APC缓存。
Forward、D prealloc/transfer仍无同口径计数，记“—”。
PPD NCCL每worker预分配32 GiB pinned pool（共256 GiB），P 1 GB/D 10 GB传输暂存额外占用HBM；
0.80/0.60是引擎配置比例，不是进程总显存硬上限。两秒采样GPU0–7峰值依次为
69,544/77,900/69,930/77,730/69,930/77,430/69,378/77,555 MiB，包含搜索和原有进程。
与NIXL baseline的传输内存开销不同，比例相同不代表总HBM开销相同。

PPD全程代理守恒4415=3903完成+512测量结束取消，failed/active为0；
此计数包含8个代理预检请求，另4个warm-P/cold-D探针直接访问引擎。
四D记录402/360/380/361次完整36层接收，对应387/353/372/350个唯一请求；重复接收含抢占恢复。
最终8引擎暂存GPU字节/retained layers/Host allocations均0、native/proxy队列均0，
自有服务全部退出，仅原有ComfyUI保留。240条未确认完成的协议记录不等同于物理泄漏，
末次暂存日志和进程退出提供收尾证据，不宣称逐条协议记录均有独立释放fence证明。
输入/运行代码/profile哈希未变；CPU回归、GPU预检和独立审计记录见原始结果。
[PPD时间序列](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/ppd_timeseries.png)
显示正式首分钟约1925 token/s，随后一分钟分桶约1112–1487 token/s；时长达标不等于已证明统计平稳。

## 新方法路径计数

统一为正式1200秒，按唯一snapshot去重；不是含预热的整轮计数，也不是Agent比例。

| 并发 | Direct完成 | Slow | 显式重算 | Direct / Slow / 重算 | D→P Host写入 / 恢复 | P→D Host写入 / 恢复 |
|---|---:|---:|---:|---|---|---|
| c384 | 5,582 | 1,503 | 136 | 77.30% / 20.81% / 1.88% | 1,503 / 1,500 | 4,349 / 4,364 |
| c512 | 5,785 | 1,126 | 244 | 80.85% / 15.74% / 3.41% | 1,126 / 1,095 | 3,090 / 3,013 |
| c576 | 5,639 | 1,070 | 306 | 80.39% / 15.25% / 4.36% | 1,070 / 1,064 | 3,845 / 3,808 |

正式窗口写入的1126个D→P snapshot中，1095个窗口内恢复、29个窗口后恢复，2个Agent因budget结束无需下一轮。
1126个均有D HBM释放记录。1次D→P Host释放、4次P→D源P释放跨窗口边界，已核对到释放。
预算结束后的Host清理及时性仍需核对，不宣称所有生命周期边界已经验收完毕。
本轮CPU回归446项通过、独立审核GO、正式请求失败0；仅一次正式结果，不代表多seed统计结论。

## 原始结果

- PPD-Dynamic c512：[summary](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/ppd_summary.json)、[数据有效性与资源](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/validation.json)、[测量窗口](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/closed_loop_boundaries.json)、[配置与代码哈希](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/reproducibility.json)、[逐token预检](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/correctness-probe.json)、[运行版本](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/vllm-ppd-packages.txt)、[协议收尾](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/proxy-status.json)、[队列排空](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/drain.json)、[GPU清理](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/gpu-processes-after-cleanup.csv)、[输入完整性](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/input-integrity-after.json)、[独立结果审计](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/RESULT_AUDIT.md)、[时间序列SVG](vllm-ppd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r2/ppd_timeseries.svg)。

- 原生 vLLM PD c512：[summary](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/vllm_summary.json)、[数据有效性与资源](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/validation.json)、[测量窗口](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/closed_loop_boundaries.json)、[配置与代码哈希](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/reproducibility.json)、[版本](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/vllm-packages.txt)、[协议收尾](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/pd-proxy-status.json)、[GPU清理](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/gpu-processes-after-cleanup.csv)、[输入完整性](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/input-integrity-after.json)、[独立结果审计](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/RESULT_AUDIT.md)、[时间序列SVG](vllm-pd/browsecomp-qwen3-8b-4p4d-c512-w300-m1200-20260917-r1/vllm_pd_timeseries.svg)。启动和适配说明见[scripts/vllm](../scripts/vllm/README_PD.md)。
- 新方法 c512：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)、[原始实验报告](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1/RESULT.md)。
- 新方法 c384：[summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)。正式窗口完成2800个Agent、请求失败0；D→P和P→D写入/恢复计数受窗口边界影响，不能直接把差额解释为丢失。
- 新方法 c576：[summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)。正式窗口完成2801个Agent、请求失败0；同样按窗口内事件独立计数，边界差额不是KV丢失量。
- Colocated c384: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c384-w300-m1200-20260816-r1/offload_analysis_summary.json)
- Colocated c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/baseline-colocated/offload_analysis_summary.json)
- Colocated c576: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c576-w300-m1200-20260817-r1/offload_analysis_summary.json)
- No-reverse PD c384: [summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- No-reverse PD c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- No-reverse PD c576: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- 原生 Mooncake c384旧轮（仅保留诊断，不计正式结果）：[summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native_mooncake-aligned-20260908-r1/offload_analysis_summary.json)。全部worker未显式绑定本地NUMA，需重跑。
- 原生 Mooncake c512旧轮（仅保留诊断，不计正式结果）：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native_mooncake-aligned-20260908-r1/offload_analysis_summary.json)。全部worker未显式绑定本地NUMA，需重跑。
- 原生 Mooncake c576（2026-09-10有效重试）：[summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/offload_analysis_summary.json)、[数据有效性及资源](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/data_validity_and_resources.json)、[实验报告及复现配置](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/RESULTS.md)。
- 原生 Mooncake c576历史失败记录保留：第一次业务预热约124秒NIXL断连，见[故障报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260909-r1/FAILURE.md)；第二次完整运行但7038条轨迹搜索失败，见[数据正确性报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260909-r3/FAILURE.md)；本次r4在业务前因NIXL初始化超过300秒watchdog退出，见[启动故障报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r4/FAILURE.md)。均不纳入正式吞吐。

消融见 [BrowseComp Qwen3-8B消融表](BROWSECOMP_QWEN3_8B_ABLATIONS.md)。

- 原生 Mooncake c384 本地NUMA重跑：[结果](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native_mooncake-numa-20260916-r1/result.json)、[summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native_mooncake-numa-20260916-r1/offload_analysis_summary.json)、[数据有效性](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native_mooncake-numa-20260916-r1/data_validity_and_resources.json)。

- 原生 Mooncake c512 本地NUMA重跑：[结果](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native_mooncake-numa-20260916-r1/result.json)、[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native_mooncake-numa-20260916-r1/offload_analysis_summary.json)、[数据有效性](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native_mooncake-numa-20260916-r1/data_validity_and_resources.json)。
