# BrowseComp Qwen3-30B-A3B TP=2 c320：P:D 比例消融

新增实验进度：`8/8`。2P:2D 固定在 `node2700`，3P:1D 固定在 `node2701`；两个节点并行、每个节点上的四种方法串行。1P:3D 直接复用 `BROWSECOMP_QWEN3_30B_A3B_TP2_H100.md` 的 c320 正式结果。

所有实验保持 BrowseComp source-order n680、temperature=0、seed=2026、context=40960、单 trajectory 预算 36864、并发 c320、300 秒预热 + 1200 秒稳态测量，以及总计 4 个 TP=2 推理组。Agentic-PD 的工具/Direct 窗口均为 1 秒，D→P Host 总量 512 GiB、P→D Host 总量 256 GiB。

> Colocated 没有 P/D 角色划分；2P:2D 与 3P:1D 两行是相同四副本配置的独立重复，用于分别控制两个比例队列的节点/时间波动。

## 吞吐与工作量

| 逻辑 P:D | 方法 | 节点 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | Prefill/agent | Decode/agent | 完成/失败 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1P:3D | Agentic-PD（新方法） | node2801（复用） | 完成 | 14,598.4 | 6.627 | 43,576.4 | 100.00% | 6,558.7 | 2,197.2 | 7,952/0 |
| 1P:3D | Colocated | node2801（复用） | 完成 | 13,500.8 | 6.037 | 45,806.8 | 82.54% | 7,561.5 | 2,228.6 | 7,245/0 |
| 1P:3D | 原生 HiCache/Mooncake PD | node2801（复用） | 完成 | 11,157.6 | 4.970 | 48,738.8 | 61.44% | 9,787.3 | 2,240.6 | 5,964/0 |
| 1P:3D | 原生单向 PD（No-reverse） | node2801（复用） | 完成 | 8,873.6 | 4.031 | 60,115.7 | 0.00% | 14,882.9 | 2,196.8 | 4,837/0 |
| 2P:2D | Agentic-PD（新方法） | node2700 | 完成 | 9,055.8 | 4.007 | 28,768.0 | 100.00% | 7,171.5 | 2,257.5 | 4,809/0 |
| 2P:2D | Colocated | node2700 | 完成 | 13,523.1 | 6.023 | 44,632.1 | 87.71% | 7,399.5 | 2,242.0 | 7,228/0 |
| 2P:2D | 原生 HiCache/Mooncake PD | node2700 | 完成 | 7,805.8 | 3.492 | 30,070.2 | 75.25% | 8,587.7 | 2,229.2 | 4,190/0 |
| 2P:2D | 原生单向 PD（No-reverse） | node2700 | 完成 | 8,533.5 | 3.842 | 59,384.4 | 0.00% | 15,429.2 | 2,217.2 | 4,610/0 |
| 3P:1D | Agentic-PD（新方法） | node2701 | 完成 | 3,911.0 | 1.748 | 12,195.0 | 100.00% | 6,959.2 | 2,231.9 | 2,098/0 |
| 3P:1D | Colocated | node2701 | 完成 | 13,493.2 | 6.007 | 44,819.3 | 85.67% | 7,442.9 | 2,240.8 | 7,209/0 |
| 3P:1D | 原生 HiCache/Mooncake PD | node2701 | 完成 | 3,659.5 | 1.635 | 13,688.9 | 74.34% | 8,349.0 | 2,232.0 | 1,962/0 |
| 3P:1D | 原生单向 PD（No-reverse） | node2701 | 完成 | 3,733.9 | 1.662 | 24,520.4 | 0.00% | 14,724.4 | 2,242.2 | 1,994/0 |

## 稳态资源

| 逻辑 P:D | 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1P:3D | Agentic-PD（新方法） | 83.3% | 15.8% | 0.63 | 10.81 | 98.4% | 79.4% | 100.90 | 0.00 | 2.08 | 0.46 |
| 1P:3D | Colocated | 22.2% | — | — | — | 77.5% | 62.0% | 78.47 | 0.88 | 0.00 | 0.00 |
| 1P:3D | 原生 HiCache/Mooncake PD | 86.1% | 4.5% | 19.44 | 1.74 | 99.5% | 78.7% | 72.17 | 0.00 | 26.01 | 8.11 |
| 1P:3D | 原生单向 PD（No-reverse） | 99.9% | 3.7% | 88.75 | 1.08 | 99.6% | 79.8% | 43.99 | 0.00 | 31.73 | 30.74 |
| 2P:2D | Agentic-PD（新方法） | 44.1% | 31.6% | 0.01 | 27.39 | 96.9% | 89.2% | 98.25 | 0.00 | 4.97 | 0.31 |
| 2P:2D | Colocated | 21.6% | — | — | — | 78.1% | 62.6% | 78.77 | 0.56 | 0.00 | 0.00 |
| 2P:2D | 原生 HiCache/Mooncake PD | 28.5% | 2.4% | 0.26 | 1.81 | 99.7% | 88.3% | 86.78 | 0.00 | 71.59 | 1.22 |
| 2P:2D | 原生单向 PD（No-reverse） | 49.4% | 2.6% | 1.11 | 1.42 | 99.6% | 86.2% | 83.72 | 0.00 | 73.36 | 2.53 |
| 3P:1D | Agentic-PD（新方法） | 16.0% | 58.0% | 0.00 | 52.61 | 97.3% | 88.5% | 69.52 | 0.00 | 3.57 | 0.23 |
| 3P:1D | Colocated | 21.9% | — | — | — | 77.8% | 62.9% | 78.77 | 0.63 | 0.00 | 0.00 |
| 3P:1D | 原生 HiCache/Mooncake PD | 9.3% | 1.3% | 0.01 | 1.38 | 99.7% | 89.7% | 62.70 | 0.00 | 255.93 | 1.05 |
| 3P:1D | 原生单向 PD（No-reverse） | 14.0% | 1.3% | 0.07 | 1.24 | 99.6% | 89.5% | 63.17 | 0.00 | 255.62 | 0.93 |

## Host 与原始目录

Agentic-PD 两种新比例都固定为 D→P 512 GiB、P→D 256 GiB 总容量。原生 HiCache/Mooncake 沿用此前 256 GiB 共享 Host cache。

- `1p3d/new_method`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c320-w300-m1200-20261012-r1`
- `1p3d/colocated`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c320-w300-m1200-20261012-r1`
- `1p3d/native_mooncake`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c320-w300-m1200-20261012-r1`
- `1p3d/no_reverse`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c320-w300-m1200-20261012-r1`
- `2p2d/new_method`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-2p2d-new-method-c320-w300-m1200-20261021-r1`
- `2p2d/colocated`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-2p2d-colocated-c320-w300-m1200-20261021-r1`
- `2p2d/native_mooncake`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-2p2d-native-mooncake-c320-w300-m1200-20261021-r1`
- `2p2d/no_reverse`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-2p2d-no-reverse-c320-w300-m1200-20261021-r1`
- `3p1d/new_method`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-3p1d-new-method-c320-w300-m1200-20261021-r1`
- `3p1d/colocated`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-3p1d-colocated-c320-w300-m1200-20261021-r1`
- `3p1d/native_mooncake`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-3p1d-native-mooncake-c320-w300-m1200-20261021-r1`
- `3p1d/no_reverse`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-3p1d-no-reverse-c320-w300-m1200-20261021-r1`
