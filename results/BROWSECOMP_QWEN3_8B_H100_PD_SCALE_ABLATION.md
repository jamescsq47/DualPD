# BrowseComp Qwen3-8B H100 P:D 等比例缩放消融

> 自动矩阵进度：**12/12**（其中 4P:4D/c512 四组复用既有有效结果）。
> 2P:2D/c256 与 1P:1D/c128 共八组在同一个 H100 allocation 内顺序重测；
> 每组采用 300 秒业务预热 + 1200 秒正式稳态窗口。除 P:D 数量与并发外，
> 模型、BrowseComp 请求序、上下文、采样、搜索服务和方法参数均保持一致。

## 配置

| 拓扑 | 并发 | P GPU | D GPU | 新方法 D→P Host | 新方法 P→D Host |
|---|---:|---|---|---:|---:|
| 4P:4D | 512 | `0 2 4 6` | `1 3 5 7` | 512 GiB total | 256 GiB total |
| 2P:2D | 256 | `0 4` | `1 5` | 512 GiB total | 256 GiB total |
| 1P:1D | 128 | `0` | `1` | 512 GiB total | 256 GiB total |

说明：原生 HiCache/Mooncake 保留其原生机制与每实例容量配置；搜索模型继续与最后一张 D/Colocated GPU 共卡。

## 核心结果

| 拓扑 | 并发 | 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 4P:4D | 512 | 当前新方法 | 复用旧结果 | 10,634.1 | 5.2117 | 78,625.9 | 95.34% | 6,254 / 0 |
| 4P:4D | 512 | Colocated | 复用旧结果 | 9,993.3 | 4.9800 | 128,014.6 | 58.30% | 5,976 / 0 |
| 4P:4D | 512 | No-reverse PD | 复用旧结果 | 4,899.7 | 2.4467 | 111,359.0 | 0.00% | 2,936 / 0 |
| 4P:4D | 512 | 原生 HiCache/Mooncake | 复用旧结果 | 5,331.9 | 2.6350 | 106,450.0 | 14.64% | 3,162 / 0 |
| 2P:2D | 256 | 当前新方法 | 完成 | 5,108.2 | 2.5358 | 37,180.3 | 97.08% | 3,043 / 0 |
| 2P:2D | 256 | Colocated | 完成 | 4,972.8 | 2.4675 | 63,734.0 | 57.13% | 2,961 / 0 |
| 2P:2D | 256 | No-reverse PD | 完成 | 2,587.5 | 1.3117 | 58,102.5 | 0.00% | 1,574 / 0 |
| 2P:2D | 256 | 原生 HiCache/Mooncake | 完成 | 3,019.9 | 1.4950 | 53,978.8 | 26.04% | 1,794 / 0 |
| 1P:1D | 128 | 当前新方法 | 完成 | 2,273.2 | 1.1133 | 16,888.1 | 95.65% | 1,336 / 0 |
| 1P:1D | 128 | Colocated | 完成 | 2,462.6 | 1.2192 | 31,522.9 | 63.37% | 1,463 / 0 |
| 1P:1D | 128 | No-reverse PD | 完成 | 1,408.4 | 0.6892 | 29,793.6 | 0.00% | 827 / 0 |
| 1P:1D | 128 | 原生 HiCache/Mooncake | 完成 | 1,576.2 | 0.7925 | 20,644.4 | 61.63% | 951 / 0 |

## 稳态资源细化

| 拓扑 | 并发 | 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4P:4D | 512 | 当前新方法 | 72.69% | 83.56% | 35.74 | 16.34 | 96.71% | 85.29% | 50.97 | 0.00 | 1.54 | 0.34 |
| 4P:4D | 512 | Colocated | 59.15% | — | — | — | 40.69% | 81.31% | 60.68 | 2.19 | — | — |
| 4P:4D | 512 | No-reverse PD | 95.37% | 7.80% | 14.47 | 0.71 | 99.35% | 91.60% | 12.90 | 0.00 | 98.18 | 16.73 |
| 4P:4D | 512 | 原生 HiCache/Mooncake | 94.23% | 8.54% | 13.34 | 0.86 | 99.37% | 91.26% | 15.92 | 0.00 | 96.00 | 15.84 |
| 2P:2D | 256 | 当前新方法 | 69.10% | 81.27% | 25.11 | 20.52 | 97.30% | 86.37% | 47.82 | 0.00 | 1.86 | 0.38 |
| 2P:2D | 256 | Colocated | 58.77% | — | — | — | 41.04% | 84.36% | 58.95 | 3.98 | 0.00 | 0.00 |
| 2P:2D | 256 | No-reverse PD | 99.98% | 8.29% | 12.48 | 0.77 | 99.36% | 91.11% | 13.78 | 0.00 | 99.01 | 15.02 |
| 2P:2D | 256 | 原生 HiCache/Mooncake | 96.33% | 9.59% | 8.55 | 0.92 | 99.46% | 89.64% | 23.32 | 0.00 | 93.07 | 11.37 |
| 1P:1D | 128 | 当前新方法 | 60.95% | 82.95% | 25.16 | 23.35 | 97.24% | 85.27% | 36.85 | 0.00 | 1.56 | 0.27 |
| 1P:1D | 128 | Colocated | 58.24% | — | — | — | 41.53% | 85.52% | 55.35 | 7.54 | 0.00 | 0.00 |
| 1P:1D | 128 | No-reverse PD | 99.84% | 8.26% | 7.66 | 0.83 | 99.34% | 89.54% | 15.65 | 0.00 | 101.97 | 10.20 |
| 1P:1D | 128 | 原生 HiCache/Mooncake | 74.80% | 9.14% | 1.46 | 1.29 | 99.52% | 87.51% | 30.59 | 0.00 | 93.30 | 3.94 |

## 原始结果目录

- `browsecomp8b-scale-4p4d-new_method-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c512-w300-m1200-20261006-r1`
- `browsecomp8b-scale-4p4d-colocated-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c512-w300-m1200-20261006-r1`
- `browsecomp8b-scale-4p4d-no_reverse-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c512-w300-m1200-20261006-r1`
- `browsecomp8b-scale-4p4d-native_mooncake-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c512-w300-m1200-20261006-r1`
- `browsecomp8b-scale-2p2d-new_method-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-2p2d-new-method-c256-w300-m1200-20261021-r1`
- `browsecomp8b-scale-2p2d-colocated-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-2p2d-colocated-c256-w300-m1200-20261021-r1`
- `browsecomp8b-scale-2p2d-no_reverse-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-2p2d-no-reverse-c256-w300-m1200-20261021-r1`
- `browsecomp8b-scale-2p2d-native_mooncake-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-2p2d-native-mooncake-c256-w300-m1200-20261021-r1`
- `browsecomp8b-scale-1p1d-new_method-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-1p1d-new-method-c128-w300-m1200-20261021-r1`
- `browsecomp8b-scale-1p1d-colocated-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-1p1d-colocated-c128-w300-m1200-20261021-r1`
- `browsecomp8b-scale-1p1d-no_reverse-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-1p1d-no-reverse-c128-w300-m1200-20261021-r1`
- `browsecomp8b-scale-1p1d-native_mooncake-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-pd-scale-1p1d-native-mooncake-c128-w300-m1200-20261021-r1`

## 最终验收

- 2026-10-22：12/12 状态均为 `complete`；各组 MD 核心指标与原始 `offload_analysis_summary.json` 逐项一致，12 个 `closed_loop_boundaries.json` 均为 `steady_churn`。
- 最终 allocation：Slurm Job `26555585`，`node2700`，状态 `COMPLETED`，退出码 `0:0`；结束后节点恢复为 `idle`，无残留实验 allocation。
- 最后一组完成状态在清理硬门禁之后写入：实验端口全部释放、GPU compute PID 归零、`apptainer`/`squashfuse_ll`/`fuse-overlayfs` helper 归零。任一条件不满足时控制器不会写入 12/12，也不会正常退出。
