# BrowseComp Qwen3-8B H100：配对 P-credit 的原生 P→D 预留消融

> 单节点 H100×8，TP=1，4P:4D，c576；300 秒业务预热 + 1200 秒稳态。
> 保留 D 原生无界 P→D preallocation；每个请求只有在取得可容纳完整工作集的 P 逻辑 credit 后才提交其 D 预留。
> parent arrival 立即通知旧 D：1 秒内目标 D 与 P credit 配对成功可走 Direct，否则先 D→Host 释放旧 D；目标 D 预留完成后才 Host→P、P Forward、P→D。

## 核心结果

| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成数 |
|---|---|---:|---:|---:|---:|---:|
| Colocated baseline（复用） | 完成 | 9,329.5 | 4.6842 | 137,779.3 | 47.08% | 5,621 |
| 完整 P→D Direct/Host 双路径（复用） | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 |
| r9 D-first、无容量 P credit（诊断） | 完成但尾段饥饿 | 3,013.6 | 1.4767 | 37,184.3 | 62.28% | 1,772 |
| r10 配对 P credit + 原生 D preallocation | 完成 | 4,601.7 | 2.2842 | 95,951.8 | 23.55% | 2,741 |

## 稳态资源细化

| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整双路径 | 73.22% | 77.40% | 51.08 | 17.30 | 97.48% | 85.07% | 52.33 | 0.00 | 1.88 | 0.36 |
| r9 无容量 credit | 32.29% | 25.74% | 0.18 | 0.94 | 83.49% | 45.32% | 7.00 | 0.00 | 105.49 | 5.91 |
| r10 配对 P credit | 84.03% | 9.53% | 7.39 | 0.97 | 99.16% | 71.38% | 11.44 | 0.00 | 23.66 | 9.49 |

## 配置与原始记录

- GPU：P=`0,2,4,6`，D=`1,3,5,7`；各 TP=1。
- D→P Host：每个 P 128 GiB，总 512 GiB；P→D Host 禁用。
- D preallocation 不设置 request/transfer 人工上限；P credit 由实时 HBM used + Router 未提交 workset 原子记账。
- 完整方法：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`
- r9：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-dfirst-recovery-after-d-direct-c576-w300-m1200-20261028-r9`
- r10：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-paired-credit-direct-c576-w300-m1200-20261028-r10`
- 控制目录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-paired-credit-c576-20261028-r10`
