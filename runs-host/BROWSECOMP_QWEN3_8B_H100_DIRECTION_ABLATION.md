# BrowseComp Qwen3-8B H100：D→P / P→D 方向消融

> 单节点 H100×8，TP=1，4P:4D，c576；固定 source-order n680，
> 300 秒业务预热 + 1200 秒稳态测量。新实验只改变 P→D：从原生
> preallocation 换为 late-binding Direct/Host 双路径；D→P 仍完全重算。

## 2×2 核心结果

| D→P parent 恢复 | P→D 路径 | 角色 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 原生（无反向恢复） | 原生 preallocate | Baseline：都不改 | 完成 | 5,297.8 | 2.6199 | 113,299.2 | 0.00% | 3,144 / 0 |
| Direct/Host | 原生 preallocate + paired P credit | 仅改 D→P | 完成 | 4,601.7 | 2.2842 | 95,951.8 | 23.55% | 2,741 / 0 |
| 原生（无反向恢复） | late-binding Direct/Host | **仅改 P→D（本轮）** | 完成 | 4,987.9 | 2.4792 | 116,502.6 | 5.23% | 2,975 / 0 |
| Direct/Host | late-binding Direct/Host | 两边都改 | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |

## 2x2 稳态资源与 KV cache hit rate

| D→P parent 恢复 | P→D late-binding | KV cache hit rate | Direct-return fraction | D KV | D running | D prealloc | D transfer | D 侧未就绪合计 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 0.00% | N/A | 91.43% | 14.28 | 112.46 | 17.06 | 129.52 |
| 开 | 关 | 23.55% | 31.19% | 71.38% | 11.44 | 23.66 | 9.49 | 33.15 |
| 关 | 开 | 5.23% | N/A | 17.94% | 12.23 | 0.07 | 0.09 | 0.16 |
| 开 | 开 | 95.52% | 78.91% | 85.07% | 52.33 | 1.88 | 0.36 | 2.24 |

> `KV cache hit rate` 采用 completion-weighted、page-aligned parent-prefix reuse 口径。
> `D prealloc` 和 `D transfer` 是每个 D 实例的稳态时间平均请求数；
> 两者之和表示已进入 D 侧但尚不能 Decode 的请求数。仅 P→D 开启时的
> 5.23% 为本地/公共 Radix parent-prefix 命中，不是 D→P 反向恢复。
> `Direct-return fraction` 只在启用 D→P 时定义：在稳态窗口内对唯一 snapshot
> 去重，以 fast-arrival 且已终结的回传结果为分母。仅 D→P 为
> 2,290 Direct / 7,342 outcomes = 31.19%；两边都开为
> 12,072 Direct / 15,298 outcomes = 78.91%。

### 证据边界

该消融直接证明了两个方向的协同收益，日志也显示仅 D→P 配置与更低的
Direct-return fraction 相关。但这些数据尚不能将差异唯一归因于额外的 P/D
admission 延迟，因为配置同时改变了请求长度分布、队列驻留时间和 HBM 状态。
同样，仅 D→P 将 D 侧未就绪请求从 129.52 降至 33.15，但 D running
也从 14.28 降至 11.44；因此 prealloc/transfer 请求数只表示调度状态深度，
不能单独代表 HBM 压力或解释全部吞吐差异。

### 仅 P→D late-binding 的稳态资源细节

| P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D prealloc | D transfer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99.65% | 8.73% | 129.30 | 0.99 | 99.34% | 17.94% | 12.23 | 0.07 | 0.09 |

## 语义门禁

- `SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true`：后续轮不携带 parent generation，D→P Direct/Host 均不可达。
- `SGLANG_PD_ABLATION_P2D_* = false`：不启用 prebind/native-prealloc/paired-credit 消融。
- `P2D_HOST_STAGING=true`：保留完整方法的 P→D Direct/Host 双路径与 late binding。
- `SGLANG_PD_LATE_BIND_MAX_DIRECT_TRANSFER_INFLIGHT=4`：Router 只在 D 的 Direct admission window 确实有空位时绑定；D 的原生总 transfer 上限仍为 8，剩余窗口供已经持久化的 P→D Host 恢复。
- 其余模型、拓扑、显存比例、Host 容量、lane、chunk、数据顺序和测量窗口与完整 c576 方法一致。

## 修复与运行审计

- r1 的 151.2 Decode token/s 是无效诊断结果：Router 在 D 已 `transfer=8/8` 时仍继续绑定 Direct，导致 P KV 长期被卡住。r3 已修复该 admission 不一致。
- r3 的 P→D Host 事件闭合：offer 774、bind 774、D2H complete 774、H2D complete 774；没有遗留 Host snapshot。
- r3 的 Direct/控制异常均为 0：NIXL sender/receiver exception 0、waiting timeout 0、reservation timeout 0、OOM 0、Traceback 0。
- D 日志观察到的最大 transfer 为 3，未越过 Router Direct 上限 4；稳态 D prealloc 0.07、D transfer 0.09。
- `Parent KV 复用=5.23%` 是 D→P 关闭后的本地/公共 Radix 前缀命中，不代表反向恢复；分析器记录的 D→P reverse reuse lower bound 仅 0.02%。
- 完成后的 node2700 清理门禁：GPU 进程 0、实验端口 0、Apptainer/FUSE helper 0、实验服务进程 0；7 天 Slurm allocation 由 holder 继续保留。

## 原始记录

- 控制目录：`/orcd/pool/004/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r3`
- 本轮结果：`/orcd/pool/004/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-only-c576-w300-m1200-20261030-r3`
- Baseline：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c576-w300-m1200-20261006-r1`
- 仅 D→P：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-paired-credit-direct-c576-w300-m1200-20261028-r10`
- 两边都改：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`
