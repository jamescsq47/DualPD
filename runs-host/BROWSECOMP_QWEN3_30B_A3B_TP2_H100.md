# BrowseComp + Qwen3-30B-A3B TP=2（单节点 H100）

测试节点：c192/c256/c320 使用 `node2801`，c128/c384 使用 `node2800`；
两者均为 8×H100 80GB。此矩阵仅把原
`BROWSECOMP_QWEN3_32B_TP2_H100.md` 的模型替换为 `Qwen3-30B-A3B`，
其余配置保持一致。当前进度：`20/20`。

BrowseComp source-order n680，temperature=0、seed=2026、context=40960、
trajectory budget=36864；每组 300 秒业务预热 + 1200 秒正式测量，
并发 c128/c192/c256/c320/c384。实验严格按表格顺序串行执行，任一组非可恢复失败时
控制器 fail-closed，不会静默跳到下一组。

PD 方法保持 2P:6D 物理布局：TP=2 Prefill `[0,4]`，TP=2 Decode
`[1,5]`、`[2,6]`、`[3,7]`。P 显存比例 0.80；D 为 0.80/0.80/0.60；
GPU7 同时承载搜索服务。Agentic-PD 使用 1 秒工具窗口、1 秒 Direct 窗口、
Q high/low=32/8，D→P Host 总量 512 GiB、P→D Host 总量 256 GiB。

## 正式结果

| 方法 | 并发 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---|---:|---|---:|---:|---:|---:|---:|
| Agentic-PD（当前新方法） | 128 | 完成 | 8,941.5 | 3.991 | 27,387.0 | 100.00% | 4,789 / 0 |
| Agentic-PD（当前新方法） | 192 | 完成 | 11,184.3 | 4.975 | 34,878.8 | 100.00% | 5,970 / 0 |
| Agentic-PD（当前新方法） | 256 | 完成 | 12,975.7 | 5.827 | 40,705.8 | 100.00% | 6,993 / 0 |
| Agentic-PD（当前新方法） | 320 | 完成 | 14,598.4 | 6.627 | 43,576.4 | 100.00% | 7,952 / 0 |
| Agentic-PD（当前新方法） | 384 | 完成 | 15,194.1 | 6.886 | 45,653.9 | 100.00% | 8,263 / 0 |
| Colocated | 128 | 完成 | 8,713.5 | 3.930 | 26,886.8 | 92.98% | 4,716 / 0 |
| Colocated | 192 | 完成 | 10,814.8 | 4.861 | 35,485.6 | 90.69% | 5,833 / 0 |
| Colocated | 256 | 完成 | 12,548.6 | 5.612 | 37,829.7 | 94.48% | 6,735 / 0 |
| Colocated | 320 | 完成 | 13,500.8 | 6.037 | 45,806.8 | 82.54% | 7,245 / 0 |
| Colocated | 384 | 完成 | 14,305.2 | 6.377 | 50,317.8 | 79.55% | 7,652 / 0 |
| 原生单向 PD（无回传） | 128 | 完成 | 7,691.2 | 3.467 | 52,333.6 | 0.00% | 4,160 / 0 |
| 原生单向 PD（无回传） | 192 | 完成 | 8,372.4 | 3.758 | 59,453.4 | 0.00% | 4,510 / 0 |
| 原生单向 PD（无回传） | 256 | 完成 | 8,290.1 | 3.719 | 59,509.2 | 0.00% | 4,463 / 0 |
| 原生单向 PD（无回传） | 320 | 完成 | 8,873.6 | 4.031 | 60,115.7 | 0.00% | 4,837 / 0 |
| 原生单向 PD（无回传） | 384 | 完成 | 8,215.2 | 3.687 | 60,216.8 | 0.00% | 4,425 / 0 |
| 原生 HiCache/Mooncake PD | 128 | 完成 | 7,752.5 | 3.472 | 21,855.4 | 99.76% | 4,166 / 0 |
| 原生 HiCache/Mooncake PD | 192 | 完成 | 9,856.1 | 4.436 | 28,395.6 | 96.95% | 5,323 / 0 |
| 原生 HiCache/Mooncake PD | 256 | 完成 | 10,556.7 | 4.715 | 37,432.9 | 84.59% | 5,658 / 0 |
| 原生 HiCache/Mooncake PD | 320 | 完成 | 11,157.6 | 4.970 | 48,738.8 | 61.44% | 5,964 / 0 |
| 原生 HiCache/Mooncake PD | 384 | 完成 | 11,518.8 | 5.154 | 54,185.7 | 50.70% | 6,185 / 0 |

## 稳态资源

| 方法 | 并发 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agentic-PD（当前新方法） | 128 | 56.7% | 5.1% | 0.03 | 2.66 | 99.6% | 36.1% | 41.36 | 0.00 | 0.12 | 0.17 |
| Agentic-PD（当前新方法） | 192 | 70.2% | 7.6% | 0.05 | 3.67 | 99.3% | 55.8% | 61.95 | 0.00 | 0.20 | 0.27 |
| Agentic-PD（当前新方法） | 256 | 78.6% | 11.9% | 0.37 | 6.93 | 99.0% | 68.8% | 81.56 | 0.00 | 1.02 | 0.35 |
| Agentic-PD（当前新方法） | 320 | 83.3% | 15.8% | 0.63 | 10.81 | 98.4% | 79.4% | 100.90 | 0.00 | 2.08 | 0.46 |
| Agentic-PD（当前新方法） | 384 | 87.3% | 25.1% | 1.96 | 17.79 | 97.1% | 88.5% | 114.74 | 0.00 | 4.81 | 0.51 |
| Colocated | 128 | 13.1% | — | — | — | 86.6% | 24.9% | 31.62 | 0.00 | 0.00 | 0.00 |
| Colocated | 192 | 17.6% | — | — | — | 82.1% | 39.8% | 47.46 | 0.01 | 0.00 | 0.00 |
| Colocated | 256 | 18.7% | — | — | — | 81.0% | 50.4% | 63.44 | 0.01 | 0.00 | 0.00 |
| Colocated | 320 | 22.2% | — | — | — | 77.5% | 62.0% | 78.47 | 0.88 | 0.00 | 0.00 |
| Colocated | 384 | 24.2% | — | — | — | 75.5% | 72.5% | 91.78 | 3.46 | 0.00 | 0.00 |
| 原生单向 PD（无回传） | 128 | 86.9% | 3.0% | 8.48 | 0.95 | 93.4% | 41.7% | 37.31 | 0.00 | 1.50 | 3.72 |
| 原生单向 PD（无回传） | 192 | 99.5% | 3.6% | 40.99 | 1.07 | 94.3% | 62.8% | 43.39 | 0.00 | 5.61 | 14.42 |
| 原生单向 PD（无回传） | 256 | 99.7% | 3.6% | 72.03 | 1.07 | 99.4% | 72.4% | 39.99 | 0.00 | 20.02 | 25.16 |
| 原生单向 PD（无回传） | 320 | 99.9% | 3.7% | 88.75 | 1.08 | 99.6% | 79.8% | 43.99 | 0.00 | 31.73 | 30.74 |
| 原生单向 PD（无回传） | 384 | 99.9% | 3.8% | 114.37 | 1.10 | 99.6% | 88.0% | 38.71 | 0.00 | 49.75 | 39.28 |
| 原生 HiCache/Mooncake PD | 128 | 48.0% | 2.0% | 0.55 | 1.45 | 94.8% | 40.5% | 38.84 | 0.00 | 2.51 | 0.77 |
| 原生 HiCache/Mooncake PD | 192 | 58.9% | 3.2% | 1.07 | 1.69 | 96.8% | 59.0% | 58.90 | 0.00 | 3.47 | 1.31 |
| 原生 HiCache/Mooncake PD | 256 | 70.7% | 3.9% | 3.35 | 1.79 | 99.1% | 67.6% | 66.49 | 0.00 | 16.11 | 2.35 |
| 原生 HiCache/Mooncake PD | 320 | 86.1% | 4.5% | 19.44 | 1.74 | 99.5% | 78.7% | 72.17 | 0.00 | 26.01 | 8.11 |
| 原生 HiCache/Mooncake PD | 384 | 93.4% | 4.7% | 27.09 | 1.55 | 99.4% | 83.4% | 75.76 | 0.00 | 41.25 | 10.54 |

资源指标均按正式 1,200 秒窗口统计；Forward 按物理 GPU，KV 与队列按
逻辑 TP=2 引擎。Colocated 的 P/D 共享 KV pool，因此独立 P 指标留空。

## 原生 Mooncake Host 状态

| 并发 | 共享容量 | 窗口末驻留 | 驻留率 | 累计淘汰 | 淘汰 key | 分配失败 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 256.0 GiB | 199.84 GiB | 78.10% | 1,007.89 GiB | 688,058 | 0 |
| 192 | 256.0 GiB | 217.15 GiB | 84.80% | 985.69 GiB | 672,908 | 0 |
| 256 | 256.0 GiB | 202.67 GiB | 79.20% | 986.40 GiB | 673,388 | 0 |
| 320 | 256.0 GiB | 206.20 GiB | 80.50% | 1,004.70 GiB | 685,883 | 0 |
| 384 | 256.0 GiB | 207.20 GiB | 80.90% | 1,006.07 GiB | 686,818 | 0 |

## 原始结果目录

- `browsecomp30ba3b-new_method-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c128-w300-m1200-20261012-r1`
- `browsecomp30ba3b-new_method-c192`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c192-w300-m1200-20261012-r1`
- `browsecomp30ba3b-new_method-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c256-w300-m1200-20261012-r1`
- `browsecomp30ba3b-new_method-c320`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c320-w300-m1200-20261012-r1`
- `browsecomp30ba3b-new_method-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-new-method-c384-w300-m1200-20261012-r1`
- `browsecomp30ba3b-colocated-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c128-w300-m1200-20261012-r1`
- `browsecomp30ba3b-colocated-c192`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c192-w300-m1200-20261012-r1`
- `browsecomp30ba3b-colocated-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c256-w300-m1200-20261012-r1`
- `browsecomp30ba3b-colocated-c320`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c320-w300-m1200-20261012-r1`
- `browsecomp30ba3b-colocated-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-colocated-c384-w300-m1200-20261012-r1`
- `browsecomp30ba3b-no_reverse-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c128-w300-m1200-20261012-r1`
- `browsecomp30ba3b-no_reverse-c192`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c192-w300-m1200-20261012-r1`
- `browsecomp30ba3b-no_reverse-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c256-w300-m1200-20261012-r1`
- `browsecomp30ba3b-no_reverse-c320`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c320-w300-m1200-20261012-r1`
- `browsecomp30ba3b-no_reverse-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-no-reverse-c384-w300-m1200-20261012-r1`
- `browsecomp30ba3b-native_mooncake-c128`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c128-w300-m1200-20261012-r1`
- `browsecomp30ba3b-native_mooncake-c192`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c192-w300-m1200-20261012-r1`
- `browsecomp30ba3b-native_mooncake-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c256-w300-m1200-20261012-r1`
- `browsecomp30ba3b-native_mooncake-c320`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c320-w300-m1200-20261012-r1`
- `browsecomp30ba3b-native_mooncake-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-30b-a3b-tp2-h100-native-mooncake-c384-w300-m1200-20261012-r1`
