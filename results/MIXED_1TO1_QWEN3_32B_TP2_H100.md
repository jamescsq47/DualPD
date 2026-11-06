# Retool + BrowseComp 1:1 + Qwen3-32B TP=2（H100）

测试节点：`node2700`；GPU：8×H100 80GB。temperature 0、seed 2026、
固定 n8192 顺序、300 秒业务预热 + 1200 秒正式测量；并发为 c256/c384/c512，
共四种方法、12 组实验。当前进度：`12/12`。

固定序列为 Retool Math / BrowseComp QA 各 4096，样本池 seed=2026、任务顺序 seed=2027；
序列 SHA256 为 `94027490e74f441311451e1f230396b12959a3d5fb006202e956197e40a913a3`。

PD 方法保持 1P:3D 逻辑布局、2P:6D 物理布局：一个 TP=2 Prefill 组 `[0,4]`，
三个 TP=2 Decode 组 `[1,5]`、`[2,6]`、`[3,7]`。P 显存比例 0.80，D 分别为
0.80/0.80/0.60；GPU7 同时承载搜索服务。当前新方法使用 Q high=32、low=8，
D→P Host 总量 512 GiB、P→D Host 总量 256 GiB（均为两个物理 P rank 合计）；
原生 Mooncake 保留原生 HiCache/Mooncake 机制。

## 正式结果

| 方法 | 并发 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---|---:|---|---:|---:|---:|---:|---:|
| 当前新方法 | 256 | 完成 | 4,985.8 | 1.264 | 6,194.5 | 98.17% | 1,517 / 0 |
| 当前新方法 | 384 | 完成 | 4,954.3 | 1.289 | 6,871.1 | 91.74% | 1,547 / 0 |
| 当前新方法 | 512 | 完成 | 4,872.3 | 1.253 | 7,741.2 | 84.83% | 1,504 / 0 |
| Colocated baseline | 256 | 完成 | 5,403.8 | 1.393 | 13,367.2 | 19.96% | 1,672 / 0 |
| Colocated baseline | 384 | 完成 | 5,372.0 | 1.377 | 13,786.1 | 13.15% | 1,652 / 0 |
| Colocated baseline | 512 | 完成 | 5,242.5 | 1.322 | 13,892.5 | 12.52% | 1,587 / 0 |
| No-reverse PD | 256 | 完成 | 4,573.8 | 1.182 | 12,926.1 | 0.00% | 1,418 / 0 |
| No-reverse PD | 384 | 完成 | 4,474.5 | 1.143 | 12,860.9 | 0.00% | 1,372 / 0 |
| No-reverse PD | 512 | 完成 | 4,603.5 | 1.177 | 12,524.6 | 0.00% | 1,412 / 0 |
| 原生 Mooncake | 256 | 完成 | 4,284.7 | 1.089 | 10,024.9 | 42.69% | 1,307 / 0 |
| 原生 Mooncake | 384 | 完成 | 4,282.2 | 1.072 | 10,762.4 | 28.05% | 1,286 / 0 |
| 原生 Mooncake | 512 | 完成 | 4,497.3 | 1.147 | 10,479.9 | 24.92% | 1,376 / 0 |

## 当前新方法路径统计

路径计数在每个新方法实验完成后，按正式 1,200 秒窗口和 snapshot/key 去重补入；
窗口首尾在途项会使写入、恢复与释放存在少量差异。

| 并发 | D→P Direct | D→P Slow Host | D→P 重算 | Direct/Slow/重算 | P→D Direct | P→D Host | Host→D 恢复 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 797 | 920 | 33 | 45.54% / 52.57% / 1.89% | 298 | 2,926 | 2,928 |
| 384 | 718 | 814 | 172 | 42.14% / 47.77% / 10.09% | 280 | 2,958 | 2,954 |
| 512 | 709 | 627 | 380 | 41.32% / 36.54% / 22.14% | 289 | 2,963 | 2,965 |

## 稳态资源

| 方法 | 并发 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前新方法 | 256 | 46.6% | 92.4% | 1.74 | 50.75 | 99.5% | 87.9% | 39.54 | 0.00 | 2.56 | 0.07 |
| 当前新方法 | 384 | 49.9% | 93.4% | 24.00 | 50.66 | 99.6% | 87.7% | 39.33 | 0.00 | 2.64 | 0.09 |
| 当前新方法 | 512 | 55.1% | 92.9% | 140.16 | 49.14 | 99.5% | 88.0% | 38.89 | 0.00 | 2.67 | 0.06 |
| Colocated baseline | 256 | 23.9% | — | — | — | 76.1% | 90.5% | 42.11 | 21.30 | 0.00 | 0.00 |
| Colocated baseline | 384 | 24.5% | — | — | — | 75.4% | 90.6% | 43.05 | 52.36 | 0.00 | 0.00 |
| Colocated baseline | 512 | 24.7% | — | — | — | 75.2% | 91.4% | 41.78 | 85.62 | 0.00 | 0.00 |
| No-reverse PD | 256 | 86.6% | 8.3% | 4.94 | 1.70 | 99.9% | 88.7% | 34.66 | 0.00 | 47.68 | 2.89 |
| No-reverse PD | 384 | 86.0% | 7.9% | 4.98 | 1.66 | 99.9% | 88.9% | 33.89 | 0.00 | 91.01 | 2.91 |
| No-reverse PD | 512 | 83.7% | 7.8% | 3.57 | 1.80 | 99.9% | 89.0% | 34.93 | 0.00 | 133.21 | 2.34 |
| 原生 Mooncake | 256 | 70.6% | 7.9% | 1.74 | 1.90 | 99.9% | 88.7% | 33.29 | 0.00 | 50.36 | 1.62 |
| 原生 Mooncake | 384 | 75.1% | 7.7% | 2.54 | 1.80 | 99.9% | 88.9% | 33.36 | 0.00 | 92.58 | 1.99 |
| 原生 Mooncake | 512 | 71.7% | 7.1% | 2.23 | 1.84 | 99.9% | 89.0% | 35.47 | 0.00 | 133.31 | 1.78 |

资源指标均按正式窗口统计；Forward 按物理 GPU，KV 与队列按逻辑 TP=2 引擎。
Colocated 的 P/D 共享同一 KV pool，因此 P KV/queue/inflight 以 `—` 表示。

## 原生 Mooncake Host 状态

原生 Mooncake 使用共享 segment，不按 D→P/P→D 拆分容量。以下为正式窗口结束时
master 汇总；累计淘汰量可超过共享 segment 容量。

| 并发 | 共享容量 | 窗口末驻留 | 驻留率 | 累计淘汰 | 淘汰 key | 分配失败 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256.0 GiB | 196.82 GiB | 76.90% | 1,015.57 GiB | 259,993 | 0 |
| 384 | 256.0 GiB | 204.28 GiB | 79.80% | 990.46 GiB | 253,566 | 0 |
| 512 | 256.0 GiB | 201.43 GiB | 78.70% | 1,014.07 GiB | 259,611 | 0 |

## 原始结果目录

- `mixed32b-new_method-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-new-method-c256-w300-m1200-20261008-r1`
- `mixed32b-new_method-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-new-method-c384-w300-m1200-20261008-r1`
- `mixed32b-new_method-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-new-method-c512-w300-m1200-20261008-r1`
- `mixed32b-colocated-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-colocated-c256-w300-m1200-20261008-r1`
- `mixed32b-colocated-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-colocated-c384-w300-m1200-20261008-r1`
- `mixed32b-colocated-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-colocated-c512-w300-m1200-20261008-r1`
- `mixed32b-no_reverse-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-no-reverse-c256-w300-m1200-20261008-r1`
- `mixed32b-no_reverse-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-no-reverse-c384-w300-m1200-20261008-r1`
- `mixed32b-no_reverse-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-no-reverse-c512-w300-m1200-20261008-r1`
- `mixed32b-native_mooncake-c256`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-native-mooncake-c256-w300-m1200-20261008-r1`
- `mixed32b-native_mooncake-c384`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-native-mooncake-c384-w300-m1200-20261008-r1`
- `mixed32b-native_mooncake-c512`：`/orcd/scratch/orcd/003/shangy/dualpd/runs/mixed1to1-qwen3-32b-tp2-h100-native-mooncake-c512-w300-m1200-20261008-r1`
