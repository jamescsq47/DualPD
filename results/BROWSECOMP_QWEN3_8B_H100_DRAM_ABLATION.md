# BrowseComp Qwen3-8B H100 DRAM消融（c512）

单节点node2700，8×H100，TP=1、4P:4D；300秒业务预热＋1200秒正式窗口。
四种配置全部重跑；普通卡mem_fraction_static=0.80、GPU7=0.60。
新方法容量单位GiB；原生HiCache参数单位十进制GB；Mooncake segment为256 GiB。

KV池占用由pool/arena已分配slot/extent字节计数直接采样，包含在途分配，不等同于唯一有效KV payload。
PSS为本组进程按比例分摊共享页后的内存占用；固定池容量、KV池已分配量和进程PSS分别记录，不混用。
每2秒保留完整Prometheus响应、分池用量、节点内存和cgroup计数。
进程PSS独立线程每60秒扫描一次，记录扫描起止时间；2秒曲线携带最近PSS及其年龄，不把它当成2秒实测。

PSS峰值是60秒采样最大值；KV池有效用量峰值是2秒采样最大值，均非瞬时硬件最高水位。
r1因同步PSS扫描拖慢采样而作废并保留原始数据；r2从本组完整300+1200秒重新开始。

| 配置 | 状态 | Decode tok/s | Parent KV复用 | KV池占用均值/峰值GiB | 进程PSS均值/峰值GiB | KV覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| 新方法512+256 GiB | complete | 10516.8 | 95.69% | 191.60 / 303.93 | 963.32 / 967.14 | 100.00% |
| 新方法256+128 GiB | complete | 10778.7 | 95.47% | 93.26 / 186.85 | 579.61 / 583.10 | 100.00% |
| HiCache P128/D64 GB +256 GiB | complete | 5555.9 | 14.62% | 920.15 / 932.04 | 996.66 / 997.36 | 100.00% |
| HiCache P96/D52 GB +256 GiB | complete | 5368.6 | 12.74% | 756.42 / 768.27 | 831.31 / 831.73 | 100.00% |

## DRAM分项

容量与占用均为GiB；占用单元格为“时间加权均值 / 2秒采样峰值”。“—”表示该方法没有此池。
Mooncake占用取master_allocated_bytes，容量取master_total_capacity_bytes，不重复累加segment指标。
分项峰值可能发生于不同时刻，不能相加当作总峰值。

| 配置 | 本地池容量 | Mooncake容量 | D→P占用 | P→D占用 | HiCache P占用 | HiCache D占用 | Mooncake占用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 新方法512+256 GiB | 768.00 | — | 19.22 / 90.35 | 172.38 / 231.97 | — | — | — |
| 新方法256+128 GiB | 384.00 | — | 18.43 / 103.85 | 74.83 / 120.13 | — | — | — |
| HiCache P128/D64 GB +256 GiB | 715.29 | 256.00 | — | — | 475.09 / 476.73 | 238.42 / 238.43 | 206.63 / 218.46 |
| HiCache P96/D52 GB +256 GiB | 551.39 | 256.00 | — | — | 355.93 / 357.47 | 193.74 / 193.75 | 206.75 / 217.56 |

HiCache本地池容量是原生按page向上对齐后的实测值；不能把十进制GB参数直接当GiB。
本次单轮结果中，新方法384 GiB相对768 GiB的Decode吞吐为1.025倍；未见减半导致吞吐下降，但没有重复试验，不能据此断言减小内存能提升性能。

## 完整窗口与采样核验

| 配置 | 业务预热秒 | 正式窗口秒 | 正式KV样本数 | KV / PSS覆盖率 | 正式请求失败 / OOM |
|---|---:|---:|---:|---:|---:|
| 新方法512+256 GiB | 300.452 | 1200.013 | 600 | 100% / 100% | 0 / 0 |
| 新方法256+128 GiB | 300.914 | 1200.000 | 600 | 100% / 100% | 0 / 0 |
| HiCache P128/D64 GB +256 GiB | 301.979 | 1200.000 | 600 | 100% / 100% | 0 / 0 |
| HiCache P96/D52 GB +256 GiB | 301.625 | 1200.000 | 600 | 100% / 100% | 0 / 0 |

已逐组复核launch_config、closed_loop_boundaries、h100_result、host_resource_summary与原始采样；重新积分KV池占用和PSS，均值/峰值与表格一致。
四份完整Prometheus gzip均可读至EOF；每个正式窗口包含21个不同的最近完成PSS快照，不把600个携带缓存值的行误当600次PSS实测。

## 原始曲线数据

- 新方法512+256 GiB：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/browsecomp8b-agentic-full-c512-w300-m1200-r2`；绘图输入`host_resources_2s.csv`，完整原始响应`raw_prometheus_2s.jsonl.gz`，分池计数`dram-pools/*.jsonl`。
- 新方法256+128 GiB：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/browsecomp8b-agentic-half-c512-w300-m1200-r2`；绘图输入`host_resources_2s.csv`，完整原始响应`raw_prometheus_2s.jsonl.gz`，分池计数`dram-pools/*.jsonl`。
- HiCache P128/D64 GB +256 GiB：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/browsecomp8b-hicache-full-c512-w300-m1200-r2`；绘图输入`host_resources_2s.csv`，完整原始响应`raw_prometheus_2s.jsonl.gz`，分池计数`dram-pools/*.jsonl`。
- HiCache P96/D52 GB +256 GiB：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/browsecomp8b-hicache-reduced-c512-w300-m1200-r2`；绘图输入`host_resources_2s.csv`，完整原始响应`raw_prometheus_2s.jsonl.gz`，分池计数`dram-pools/*.jsonl`。

聚合曲线使用host_resources_2s.csv，单池曲线使用host_pools_2s.csv，实际PSS扫描起止及原始进程数据见process_memory_60s.jsonl。
第四组已由用户确认改成P96 GB×4+D52 GB×4+Mooncake256 GiB；52 GB经原生对齐为352704 tokens，严格大于GPU池345920 tokens；51 GB对齐后恰好相等，不能使用。未修改原生容量断言。

## 最终清理与释放核验

- 控制器仅在每组GPU进程为0、配置端口连续三次可绑定、Apptainer/squashfuse_ll/fuse-overlayfs辅助进程为0后才写入complete；四组state均为complete。
- 退出前独立watchdog本机探针（2026-10-22 12:04:21 UTC）：GPU进程0、Apptainer/FUSE辅助进程0；其监听端口与全部25个实验配置端口的交集为空。
- 最终heartbeat为complete、total=4。Slurm作业26573936正常COMPLETED，ExitCode=0:0；watchdog已报告4/4并退出。
- 2026-10-22 12:14 UTC调度器显示node2700为IDLE、无活动作业。上述GPU/端口/辅助进程证据来自allocation释放前的本机检查，不冒充释放后的新一次SSH检查。

<!-- DRAM_P64D64_START -->
## 追加：HiCache P64/D64（c512）

节点：node1701。TP=1、4P:4D；P64 GB×4+D64 GB×4+Mooncake256 GiB。
复用上述原生baseline冻结快照、数据、GPU布局及显存比例，仅将P池改成64 GB；不覆盖原四组。
300秒业务预热＋1200秒正式窗口；KV/节点/cgroup每2秒，PSS独立每60秒采样。

| 配置 | 状态 | Decode tok/s | Parent KV复用 | KV池占用均值/峰值GiB | 进程PSS均值/峰值GiB | KV / PSS覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| HiCache P64/D64 GB +256 GiB | complete | 2378.0 | 13.86% | 556.70 / 671.85 | 759.50 / 760.34 | 100.00% / 100.00% |

原始记录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-p64d64-c512-20261022/browsecomp8b-hicache-p64d64-c512-w300-m1200-r1`。绘图输入`host_resources_2s.csv`、`host_pools_2s.csv`；PSS实际扫描见`process_memory_60s.jsonl`；完整指标见`raw_prometheus_2s.jsonl.gz`。

实测预热300.285秒、正式1200.179秒；正式请求失败0次。
P池均值/峰值233.99 / 238.17 GiB；D池116.08 / 221.73 GiB；Mooncake206.63 / 217.65 GiB。

统计使用P/D八个端点均完整的共同采样帧，缺失端点不补零；原始metrics未改写。
累计计数覆盖1193.763秒，末120秒完整采样间Decode增加203111 tokens；旧错误聚合结果另存legacy-aggregate文件。
引擎完整帧是不规则采样，最大间隔34.219秒；派生引擎吞吐曲线不视为2秒实测，Host内存曲线仍为真实2秒采样。

complete仅在本组GPU=0、实验端口连续三次空闲、Apptainer/FUSE=0后写入。
<!-- DRAM_P64D64_END -->

<!-- DRAM_D64P128_START -->
## 额外实验：误将64/128理解为每P容量，全节点256＋512 GiB（非用户指定配置）

节点：node1701。Qwen3-8B、TP=1、4P:4D、c512；每P的D→P池64 GiB、P→D池128 GiB。
全节点D→P 256 GiB＋P→D 512 GiB，总计768 GiB。复用原新方法冻结引擎和记录的启动环境，
仅改两方向池容量和本组记录路径；普通GPU显存比例0.80，GPU7为0.60。
完整300秒预热＋1200秒稳态；Host池/节点/cgroup每2秒采样，进程PSS独立每60秒扫描。

| 配置 | 状态 | Decode tok/s | Parent KV复用 | KV池已分配量均值/峰值GiB | 整组PSS均值/峰值GiB | KV/PSS覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| 新方法256+512 GiB | complete | 10145.7 | 95.46% | 301.97 / 428.41 | 962.76 / 966.77 | 100.00% / 100.00% |

原始记录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-d64p128-c512-20261028/browsecomp8b-agentic-d64p128-c512-w300-m1200-r2`。曲线：`host_resources_2s.csv`、`host_pools_2s.csv`；实际PSS扫描：`process_memory_60s.jsonl`。
固定池容量、池内已分配量和整组PSS不是同一指标；PSS是60秒非原子扫描，峰值仅为采样峰值。

D→P已分配量均值/峰值：25.26 / 125.04 GiB；P→D：276.71 / 387.10 GiB。
实际预热301.619秒、稳态1200.001秒；请求失败0次；OOM增量见h100_result.json。
吞吐使用P/D全部八端点完整共同帧；缺失端点不补零，原始metrics保留不变。

complete前已确认GPU进程=0、实验端口连续三次空闲、Apptainer/FUSE辅助进程=0。
<!-- DRAM_D64P128_END -->

追加组重试记录：r1在预热阶段出现P-accepted超时和NIXL断连/source KV隔离，未完成300+1200窗口，不计入性能结果；原始文件完整保留。2026-10-28以相同配置启动独立r2（node1701，作业27107001），重新通过127项回归与GPU/端口/Apptainer-FUSE初始清理门禁。本组watchdog改用只读SSH探针，未修改共享引擎；断连首因尚未确定，不将干净重启表述为根因修复。

<!-- DRAM_GLOBAL64_128_START -->
## 用户指定配置：全节点 D→P 64＋P→D 128 GiB（c512）

此前256+512 GiB是将用户的全节点64/128误解为每P容量而产生的额外实验，不是本组结果。
节点：node1701；Qwen3-8B、TP=1、4P:4D、c512。
**全节点D→P=64 GiB、P→D=128 GiB，总计192 GiB；每P分别16/32 GiB。**
沿用之前Host参数的GiB计量口径；GPU布局P=0/2/4/6、D=1/3/5/7；普通卡0.80，GPU7为0.60。
其余参数、冻结SGLang、数据与采样口径保持原样；300秒预热＋1200秒稳态，Host池/节点/cgroup每2秒，PSS每60秒。

| 全节点D→P＋P→D | 每P D→P/P→D | 状态 | Decode tok/s | Parent KV复用 | KV已分配量均值/峰值GiB | 整组PSS均值/峰值GiB |
|---|---|---|---:|---:|---:|---:|
| 64＋128 GiB | 16/32 GiB | complete | 10613.5 | 94.58% | 89.70 / 166.84 | 384.72 / 386.95 |

原始记录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-global64-128-c512-20261028/browsecomp8b-agentic-global64-128-c512-w300-m1200-r1`；曲线`host_resources_2s.csv`、`host_pools_2s.csv`，PSS扫描`process_memory_60s.jsonl`。
固定容量、池内已分配量、整组PSS分开统计；PSS为非原子60秒扫描，峰值为采样峰值。

D→P已分配量均值/峰值16.62 / 54.19 GiB；P→D 73.08 / 116.66 GiB。
预热301.128秒、稳态1200.001秒；请求失败0次；KV/PSS覆盖率100.00%/100.00%。
正式窗口逐条核验四个D→P池合计64 GiB、四个P→D池合计128 GiB，不能以每P容量代替总量。

完成前已通过GPU进程=0、配置端口连续三次空闲、Apptainer/FUSE=0门禁。
<!-- DRAM_GLOBAL64_128_END -->
