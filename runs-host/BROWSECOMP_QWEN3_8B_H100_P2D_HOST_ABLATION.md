# BrowseComp Qwen3-8B H100：仅关闭 P→D Host 消融

> 单节点 H100×8，TP=1，4P:4D，c576，300 秒业务预热 + 1200 秒稳态。
> D→P Direct/Host、Slow 拥堵重算与 P→D late binding 均保持不变；
> 唯一变化为关闭 P→D Host staging。未启用 P→D prebind/native-prealloc。

## 核心结果

| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---|---|---:|---:|---:|---:|---:|
| 完整 P→D Direct/Host 双路径（复用） | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |
| 仅关闭 P→D Host | 完成 | 10,510.5 | 5.2592 | 78,704.8 | 95.93% | 6,311 / 0 |

- Direct-only / 完整双路径吞吐比：`0.975×`。
- 完整双路径相对仅关闭 P→D Host 的 Decode 吞吐提升为 **2.55%**；Agent/s 提升 **0.30%**。
- Parent KV 复用率为 95.93%（完整方法 95.52%，相差 +0.41 pp），说明本轮没有把 D→P 恢复耦合进 P→D 消融。

## 稳态资源

| 方法 | P Forward/卡 | P KV | P queue | P inflight | D Forward/卡 | D KV | D running | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整双路径 | 73.22% | 77.40% | 51.08 | 17.30 | 97.48% | 85.07% | 52.33 | 1.88 | 0.36 |
| 仅关闭 P→D Host | 72.74% | 76.69% | 72.78 | 14.85 | 97.74% | 83.70% | 50.80 | 1.56 | 0.28 |

## 有效性与清理验收

- 实际业务预热 **300.723 秒**，正式测量 **1200.001 秒**；完成 **6,311** 个 Agent，失败 **0**。
- 最后 120 秒仍产生 **1,303,861** 个 Decode tokens，没有尾段停滞。
- P→D Host staging 两个开关均为 `false`，日志中 P→D Host 写入/恢复事件为 **0**；P→D prebind 为 `false`，D→P reuse 保持开启。
- `memory.events` 的 `oom`、`oom_kill`、`high`、`max` 增量均为 **0**。
- 控制器在写入 `complete` 前通过清理硬门禁：GPU计算进程归零、全部配置端口连续三次可绑定、Apptainer/squashfuse_ll/fuse-overlayfs helper归零。
- Slurm作业 `27117195` 在 node1701 以 `COMPLETED / ExitCode 0:0` 退出，allocation已经释放。

## 原始记录

- 控制目录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-host-off-c576-20261028-r1`
- 本轮结果：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-host-off-c576-w300-m1200-20261028-r1`
- 完整方法：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`
