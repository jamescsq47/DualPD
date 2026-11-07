# BrowseComp Qwen3-8B H100：P→D 双路径消融

> 配置：单节点 H100×8，TP=1，4P:4D，c576，300 秒业务预热 + 1200 秒稳态测量。
> 正式消融先让 D 实际占用 request slot 和物理目标 KV；随后才向选定 P 发布 targeted arrival，完成既有 D→P Direct/Host 恢复并提交 P forward。P→D 只走 Direct。
> 不设置请求数或 transfer inflight 上限（两个上限参数均为 0）；仅由 D 原生 KV allocator、request slots 与 decode safety headroom 限制。
> 旧的 Router 中心预留实验改变了原生调度语义并在高并发下停滞，只保留审计，不作为性能结论。

## 核心结果

| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成数 |
|---|---|---:|---:|---:|---:|---:|
| Colocated baseline（复用） | 完成 | 9,329.5 | 4.6842 | 137,779.3 | 47.08% | 5,621 |
| 完整 P→D Direct/Host 双路径（复用） | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 |
| D 实际预留后放行 P + Direct-only（无人工上限） | 完成（尾段近停滞） | 3,013.6 | 1.4767 | 37,184.3 | 62.28% | 1,772 |
| r8 P 提前 bootstrap、恢复早于 D 预留（闭环，无效） | 无效 | 459.0 | 0.1725 | 3,374.6 | 78.00% | 207 |
| r7 D 预留后才提交 P（超时，无效） | 无效 | 257.5 | 0.0875 | 2,252.4 | 66.06% | 105 |
| r6 targeted-arrival 顺序错误（无效） | 无效 | 3,625.0 | 1.8042 | 77,792.1 | 11.85% | 2,165 |
| 旧 4/D 路由稳定运行（诊断） | 完成（尾段近停滞） | 392.4 | 0.1400 | 2,263.3 | 75.77% | 168 |
| 4/D 修复未生效运行（诊断） | 无效 | — | — | — | — | — |
| 有界原生 prealloc + Direct-only（8/D，诊断） | 失败（重定向残留） | 385.7 | 0.1408 | 2,068.7 | 91.08% | 169 |
| 无界原生 prealloc + Direct-only（诊断） | 完成（资源闭环拥堵） | 403.7 | 0.1525 | 2,522.7 | 83.92% | 183 |
| 旧中心预留实现（无效，仅审计） | 无效 | 255.2 | 0.0667 | 594.5 | 51.64% | 80 |

## 正式对比

- 完整双路径相对 Colocated：`1.155×`。
- D-first 无界消融相对 Colocated：`0.323×`。
- 完整双路径相对 D-first 无界消融的吞吐提升：`257.67%`（即 `3.577×`）。
- 完整窗口数据可复核，但正式消融最后 120 秒仅增长 `566` 个 Decode tokens，属于尾段近停滞；该数值不能当作健康原生 SGLang baseline。

- 8/D 诊断运行虽限制了 transfer，但动态 P 重定向仍触发大量 receiver abort，不能代表合理的经典预分配实现。
- 无界原生窗口的诊断值为 403.7 decode token/s；其 P/D HBM 被在途 transfer 同时钉住，不能代表合理的经典预分配实现。

## 稳态资源细化

| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 63.24% | — | — | — | 36.62% | 86.75% | 66.64 | 4.23 | — | — |
| 完整双路径 | 73.22% | 77.40% | 51.08 | 17.30 | 97.48% | 85.07% | 52.33 | 0.00 | 1.88 | 0.36 |
| D 实际预留后放行 P + Direct-only（无人工上限） | 32.29% | 25.74% | 0.18 | 0.94 | 83.49% | 45.32% | 7.00 | 0.00 | 105.49 | 5.91 |
| r8 P 提前 bootstrap、恢复早于 D 预留（闭环，无效） | — | — | — | — | — | — | — | — | — | — |
| r7 D 预留后才提交 P（超时，无效） | — | — | — | — | — | — | — | — | — | — |
| r6 targeted-arrival 顺序错误（无效） | 66.46% | 6.72% | 15.26 | 0.96 | 98.73% | 85.08% | 8.48 | 0.00 | 117.59 | 16.58 |
| 旧 4/D 路由稳定运行 | 1.76% | 96.86% | 3.19 | 1.20 | 7.91% | 17.77% | 1.97 | 0.00 | 130.33 | 4.00 |
| 4/D 修复未生效运行 | — | — | — | — | — | — | — | — | — | — |
| 有界原生 prealloc + Direct-only（8/D） | — | — | — | — | — | — | — | — | — | — |
| 无界原生 prealloc + Direct-only | 1.95% | 98.14% | 31.41 | 1.16 | 19.76% | 91.99% | 1.65 | 0.00 | 111.75 | 33.04 |

## 固定配置与原始记录

- GPU 映射：P=`0,2,4,6`，D=`1,3,5,7`；各引擎 TP=1。
- Host 容量：D→P 每个 P 128 GiB（总 512 GiB）；正式消融禁用 P→D Host。
- 调度语义：Router 先 POST D，等待 D worker 完成真实物理 KV 预留并写出逐请求证明；随后才向目标 P 发布 targeted arrival，等待 D→P 恢复确定后提交 P forward。
- 上限语义：`SGLANG_PD_MAX_TRANSFER_INFLIGHT=0`、`PD_MAX_TRANSFER_INFLIGHT=0`，无人工请求数上限。
- Colocated：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c576-w300-m1200-20261006-r1`
- 完整双路径：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`
- 正式 D-first 无界消融：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-dfirst-recovery-after-d-direct-c576-w300-m1200-20261028-r9`
- r8 恢复早于 D 预留的资源闭环运行：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-dfirst-queued-direct-c576-w300-m1200-20261028-r8`
- r7 预留后才提交 P 的超时运行：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-dfirst-targeted-direct-c576-w300-m1200-20261028-r7`
- r6 顺序错误无效运行：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-dfirst-unbounded-direct-c576-w300-m1200-20261028-r6`
- 旧 4/D 路由稳定诊断：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-settled-prealloc4-direct-c576-w300-m1200-20261028-r5`
- r4 修复未生效诊断：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-settled-prealloc4-direct-c576-w300-m1200-20261028-r4`
- 8/D 重定向残留诊断：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-bounded-prealloc8-direct-c576-w300-m1200-20261028-r3`
- 无界原生诊断：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-native-prealloc-direct-c576-w300-m1200-20261027-r2`
- 无效旧实现：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-prealloc-direct-c576-w300-m1200-20261027-r1`
- 控制状态：`/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-dfirst-recovery-after-d-c576-20261028-r9`
