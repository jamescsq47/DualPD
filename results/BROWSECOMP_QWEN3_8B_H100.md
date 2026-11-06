# BrowseComp + Qwen3-8B（H100）

有效结果分两批完成：c384/c512/c576 于 2026-10-06 在 `node1701` 测量，
c64/c128/c256 于 2026-10-18 在 `node2800` 测量；两批均使用 8×H100，共 24 组。

统一使用固定 source-order n680、seed 2026、temperature 0、top_p 1、top_k -1、
context 40960、轨迹预算 36864，以及 300 秒业务预热 + 1200 秒正式测量。
PD 配置为 4P:4D，P=0/2/4/6，D=1/3/5/7；显存比例为普通卡 0.80、GPU7 0.60。
不包含 vLLM 和 PPD。

新方法统一使用 D→P Host 总量 512 GiB（128 GiB/P）和 P→D Host 总量
256 GiB（64 GiB/P），共 768 GiB 注册窗口；使用 2 条 D2H lane、2 条 H2D lane
和 4096-token chunk，并在全部 Host 注册完成后开始业务预热。原生 Mooncake 保留
256 GB 原生共享 segment/HiCache 机制，不按方向拆分容量。

所有结果的边界状态均为 `steady_churn`，正式测量均不少于 1200 秒，请求失败数均为 0。

## 统一结果总表

| 并发 | 方法 | 节点 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |
|---:|---|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法 | node2800 | 5,623.2 | 2.8133 | 39,908.0 | 100.00% | 3,376 / 0 |
| 64 | Colocated baseline | node2800 | 5,880.8 | 2.9492 | 41,596.1 | 95.13% | 3,539 / 0 |
| 64 | No-reverse PD | node2800 | 3,453.5 | 1.7233 | 78,385.3 | 0.00% | 2,068 / 0 |
| 64 | 原生 Mooncake | node2800 | 4,108.7 | 2.0267 | 57,749.6 | 55.27% | 2,432 / 0 |
| 128 | 当前新方法 | node2800 | 8,572.2 | 4.2242 | 57,288.5 | 100.00% | 5,069 / 0 |
| 128 | Colocated baseline | node2800 | 8,575.9 | 4.2750 | 60,773.0 | 96.17% | 5,130 / 0 |
| 128 | No-reverse PD | node2800 | 4,441.3 | 2.2017 | 101,995.7 | 0.00% | 2,642 / 0 |
| 128 | 原生 Mooncake | node2800 | 5,152.3 | 2.5617 | 82,376.8 | 40.22% | 3,074 / 0 |
| 256 | 当前新方法 | node2800 | 11,262.0 | 5.5875 | 75,079.3 | 100.00% | 6,705 / 0 |
| 256 | Colocated baseline | node2800 | 10,259.2 | 5.0725 | 92,472.0 | 83.31% | 6,087 / 0 |
| 256 | No-reverse PD | node2800 | 4,896.5 | 2.4275 | 111,881.5 | 0.00% | 2,913 / 0 |
| 256 | 原生 Mooncake | node2800 | 5,721.7 | 2.8458 | 103,945.4 | 25.24% | 3,415 / 0 |
| 384 | 当前新方法 | node1701 | 10,719.7 | 5.3249 | 74,372.2 | 98.51% | 6,390 / 0 |
| 384 | Colocated baseline | node1701 | 10,622.4 | 5.3133 | 106,248.2 | 76.26% | 6,376 / 0 |
| 384 | No-reverse PD | node1701 | 4,877.0 | 2.4100 | 112,429.5 | 0.00% | 2,892 / 0 |
| 384 | 原生 Mooncake | node1701 | 5,417.8 | 2.6742 | 105,482.5 | 17.11% | 3,209 / 0 |
| 512 | 当前新方法 | node1701 | 10,634.1 | 5.2117 | 78,625.9 | 95.34% | 6,254 / 0 |
| 512 | Colocated baseline | node1701 | 9,993.3 | 4.9800 | 128,014.6 | 58.30% | 5,976 / 0 |
| 512 | No-reverse PD | node1701 | 4,899.7 | 2.4467 | 111,359.0 | 0.00% | 2,936 / 0 |
| 512 | 原生 Mooncake | node1701 | 5,331.9 | 2.6350 | 106,450.0 | 14.64% | 3,162 / 0 |
| 576 | 当前新方法 | node1701 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |
| 576 | Colocated baseline | node1701 | 9,329.5 | 4.6842 | 137,779.3 | 47.08% | 5,621 / 0 |
| 576 | No-reverse PD | node1701 | 5,297.8 | 2.6199 | 113,299.2 | 0.00% | 3,144 / 0 |
| 576 | 原生 Mooncake | node1701 | 5,358.5 | 2.6508 | 107,662.5 | 13.28% | 3,181 / 0 |

## Decode 吞吐横向对比

| 并发 | 当前新方法 | Colocated | No-reverse | 原生 Mooncake | 新方法 / Colocated | 新方法 / No-reverse | 新方法 / Mooncake |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5,623.2 | 5,880.8 | 3,453.5 | 4,108.7 | 0.956× | 1.628× | 1.369× |
| 128 | 8,572.2 | 8,575.9 | 4,441.3 | 5,152.3 | 1.000× | 1.930× | 1.664× |
| 256 | 11,262.0 | 10,259.2 | 4,896.5 | 5,721.7 | 1.098× | 2.300× | 1.968× |
| 384 | 10,719.7 | 10,622.4 | 4,877.0 | 5,417.8 | 1.009× | 2.198× | 1.979× |
| 512 | 10,634.1 | 9,993.3 | 4,899.7 | 5,331.9 | 1.064× | 2.170× | 1.994× |
| 576 | 10,778.5 | 9,329.5 | 5,297.8 | 5,358.5 | 1.155× | 2.035× | 2.011× |

## 正式窗口 token 构成

Prefill/Agent 是正式窗口内物理计算 token 除以完成 Agent 数；Decode/Agent 是正式窗口内
Decode token 总量除以完成 Agent 数，不是仅对窗口内结束轨迹重新统计的平均输出长度。

| 并发 | 方法 | Prefill/Agent tokens | Decode/Agent tokens | Parent KV 复用 |
|---:|---|---:|---:|---:|
| 64 | 当前新方法 | 14,177 | 1,998 | 100.00% |
| 64 | Colocated baseline | 14,095 | 1,993 | 95.13% |
| 64 | No-reverse PD | 45,429 | 2,002 | 0.00% |
| 64 | 原生 Mooncake | 28,443 | 2,024 | 55.27% |
| 128 | 当前新方法 | 13,552 | 2,028 | 100.00% |
| 128 | Colocated baseline | 14,204 | 2,004 | 96.17% |
| 128 | No-reverse PD | 46,196 | 2,012 | 0.00% |
| 128 | 原生 Mooncake | 32,060 | 2,005 | 40.22% |
| 256 | 当前新方法 | 13,405 | 2,011 | 100.00% |
| 256 | Colocated baseline | 18,190 | 2,018 | 83.31% |
| 256 | No-reverse PD | 45,956 | 2,011 | 0.00% |
| 256 | 原生 Mooncake | 36,453 | 2,007 | 25.24% |
| 384 | 当前新方法 | 13,951 | 2,011 | 98.51% |
| 384 | Colocated baseline | 19,960 | 1,996 | 76.26% |
| 384 | No-reverse PD | 46,551 | 2,019 | 0.00% |
| 384 | 原生 Mooncake | 39,375 | 2,022 | 17.11% |
| 512 | 当前新方法 | 15,045 | 2,035 | 95.34% |
| 512 | Colocated baseline | 25,654 | 2,003 | 58.30% |
| 512 | No-reverse PD | 45,385 | 1,997 | 0.00% |
| 512 | 原生 Mooncake | 40,381 | 2,023 | 14.64% |
| 576 | 当前新方法 | 15,013 | 2,041 | 95.52% |
| 576 | Colocated baseline | 29,337 | 1,987 | 47.08% |
| 576 | No-reverse PD | 43,138 | 2,017 | 0.00% |
| 576 | 原生 Mooncake | 40,544 | 2,018 | 13.28% |

Parent KV 复用按正式窗口内完成轨迹的完整 page-aligned 父前缀统计。

## 稳态资源细化

资源为正式窗口内各引擎采样均值。Forward 是累计 GPU execution counter 换算出的每物理卡
时间占比；KV、queue、inflight、running、prealloc 和 transfer 均为每引擎平均值。

### Prefill 侧

| 并发 | 方法 | Prefill token/s | P Forward/卡 | P KV/引擎 | P queue | P inflight |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 当前新方法 | 39,908.0 | 36.20% | 6.77% | 0.00 | 1.36 |
| 64 | Colocated baseline | 41,596.1 | 19.57% | — | — | — |
| 64 | No-reverse PD | 78,385.3 | 66.87% | 6.85% | 5.32 | 0.95 |
| 64 | 原生 Mooncake | 57,749.6 | 51.81% | 8.19% | 1.79 | 1.27 |
| 128 | 当前新方法 | 57,288.5 | 52.62% | 9.90% | 0.02 | 1.76 |
| 128 | Colocated baseline | 60,773.0 | 28.93% | — | — | — |
| 128 | No-reverse PD | 101,995.7 | 87.09% | 7.80% | 10.57 | 0.85 |
| 128 | 原生 Mooncake | 82,376.8 | 73.52% | 8.60% | 7.23 | 1.10 |
| 256 | 当前新方法 | 75,079.3 | 69.80% | 24.81% | 0.20 | 4.55 |
| 256 | Colocated baseline | 92,472.0 | 43.32% | — | — | — |
| 256 | No-reverse PD | 111,881.5 | 95.49% | 8.03% | 14.37 | 0.82 |
| 256 | 原生 Mooncake | 103,945.4 | 92.06% | 9.10% | 10.75 | 0.95 |
| 384 | 当前新方法 | 74,372.2 | 69.55% | 81.28% | 7.91 | 13.73 |
| 384 | Colocated baseline | 106,248.2 | 49.66% | — | — | — |
| 384 | No-reverse PD | 112,429.5 | 96.49% | 8.05% | 14.69 | 0.74 |
| 384 | 原生 Mooncake | 105,482.5 | 93.53% | 8.76% | 12.72 | 0.90 |
| 512 | 当前新方法 | 78,625.9 | 72.69% | 83.56% | 35.74 | 16.34 |
| 512 | Colocated baseline | 128,014.6 | 59.15% | — | — | — |
| 512 | No-reverse PD | 111,359.0 | 95.37% | 7.80% | 14.47 | 0.71 |
| 512 | 原生 Mooncake | 106,450.0 | 94.23% | 8.54% | 13.34 | 0.86 |
| 576 | 当前新方法 | 79,293.4 | 73.22% | 77.40% | 51.08 | 17.30 |
| 576 | Colocated baseline | 137,779.3 | 63.24% | — | — | — |
| 576 | No-reverse PD | 113,299.2 | 96.58% | 7.76% | 14.67 | 0.73 |
| 576 | 原生 Mooncake | 107,662.5 | 95.14% | 8.80% | 13.45 | 0.89 |

### Decode 侧

| 并发 | 方法 | Decode token/s | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 当前新方法 | 5,623.2 | 99.31% | 25.33% | 14.53 | 0.00 | 0.07 | 0.20 |
| 64 | Colocated baseline | 5,880.8 | 80.00% | 11.87% | 7.67 | 0.00 | — | — |
| 64 | No-reverse PD | 3,453.5 | 93.11% | 43.02% | 8.28 | 0.00 | 0.78 | 6.85 |
| 64 | 原生 Mooncake | 4,108.7 | 91.08% | 35.98% | 12.17 | 0.00 | 0.45 | 3.30 |
| 128 | 当前新方法 | 8,572.2 | 99.26% | 48.79% | 29.58 | 0.00 | 0.12 | 0.28 |
| 128 | Colocated baseline | 8,575.9 | 70.68% | 23.32% | 15.43 | 0.01 | — | — |
| 128 | No-reverse PD | 4,441.3 | 99.23% | 72.95% | 11.07 | 0.00 | 8.04 | 12.73 |
| 128 | 原生 Mooncake | 5,152.3 | 99.09% | 65.50% | 15.33 | 0.00 | 7.07 | 9.40 |
| 256 | 当前新方法 | 11,262.0 | 98.45% | 85.68% | 56.30 | 0.00 | 1.40 | 0.39 |
| 256 | Colocated baseline | 10,259.2 | 56.41% | 44.90% | 31.03 | 0.11 | — | — |
| 256 | No-reverse PD | 4,896.5 | 99.26% | 90.62% | 12.77 | 0.00 | 34.27 | 16.76 |
| 256 | 原生 Mooncake | 5,721.7 | 99.40% | 86.96% | 19.89 | 0.00 | 30.42 | 13.34 |
| 384 | 当前新方法 | 10,719.7 | 97.21% | 86.22% | 51.94 | 0.00 | 1.48 | 0.38 |
| 384 | Colocated baseline | 10,622.4 | 49.96% | 63.83% | 46.29 | 0.49 | — | — |
| 384 | No-reverse PD | 4,877.0 | 99.21% | 91.59% | 12.74 | 0.00 | 66.02 | 17.06 |
| 384 | 原生 Mooncake | 5,417.8 | 99.41% | 91.08% | 17.18 | 0.00 | 63.32 | 15.27 |
| 512 | 当前新方法 | 10,634.1 | 96.71% | 85.29% | 50.97 | 0.00 | 1.54 | 0.34 |
| 512 | Colocated baseline | 9,993.3 | 40.69% | 81.31% | 60.68 | 2.19 | — | — |
| 512 | No-reverse PD | 4,899.7 | 99.35% | 91.60% | 12.90 | 0.00 | 98.18 | 16.73 |
| 512 | 原生 Mooncake | 5,331.9 | 99.37% | 91.26% | 15.92 | 0.00 | 96.00 | 15.84 |
| 576 | 当前新方法 | 10,778.5 | 97.48% | 85.07% | 52.33 | 0.00 | 1.88 | 0.36 |
| 576 | Colocated baseline | 9,329.5 | 36.62% | 86.75% | 66.64 | 4.23 | — | — |
| 576 | No-reverse PD | 5,297.8 | 99.34% | 91.43% | 14.28 | 0.00 | 112.46 | 17.06 |
| 576 | 原生 Mooncake | 5,358.5 | 99.34% | 91.28% | 16.11 | 0.00 | 111.62 | 16.03 |

Colocated 的 P 和 D 数据来自同一组 8 个引擎：P Forward 表示 `forward_extend` 时间占比，
D Forward 表示 `forward_decode` 时间占比，KV 池和调度队列由二者共享。因此共享 KV、
running 和 queue 只列在 D 侧，P KV/queue/inflight 以及 D prealloc/transfer 记为“—”。

### 原生 Mooncake Host 缓存与淘汰

| 并发 | Parent KV 复用 | Resident / Capacity | Resident 占比 | 淘汰成功/尝试 | Evicted keys | Evicted GiB | 分配失败 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 55.27% | 211.68 / 256 GiB | 82.70% | 46 / 46 | 229,692 | 1,009.35 | 0 |
| 128 | 40.22% | 207.87 / 256 GiB | 81.20% | 45 / 45 | 224,670 | 987.28 | 0 |
| 256 | 25.24% | 206.49 / 256 GiB | 80.70% | 46 / 46 | 230,146 | 1,011.35 | 0 |
| 384 | 17.11% | 212.90 / 256 GiB | 83.20% | 45 / 45 | 225,405 | 990.51 | 0 |
| 512 | 14.64% | 215.27 / 256 GiB | 84.10% | 46 / 46 | 229,453 | 1,008.30 | 0 |
| 576 | 13.28% | 196.66 / 256 GiB | 76.80% | 46 / 46 | 229,631 | 1,009.09 | 0 |

所有 Mooncake 组的 eviction high watermark 均为 0.85。

## 有效结果目录

| 并发 | 方法 | 结果目录 |
|---:|---|---|
| 64 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c64-w300-m1200-20261018-r1` |
| 64 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c64-w300-m1200-20261018-r1` |
| 64 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c64-w300-m1200-20261018-r1` |
| 64 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c64-w300-m1200-20261018-r1` |
| 128 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c128-w300-m1200-20261018-r1` |
| 128 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c128-w300-m1200-20261018-r1` |
| 128 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c128-w300-m1200-20261018-r1` |
| 128 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c128-w300-m1200-20261018-r1` |
| 256 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c256-w300-m1200-20261018-r1` |
| 256 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c256-w300-m1200-20261018-r1` |
| 256 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c256-w300-m1200-20261018-r1` |
| 256 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c256-w300-m1200-20261018-r1` |
| 384 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c384-w300-m1200-20261006-r2` |
| 384 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c384-w300-m1200-20261006-r1` |
| 384 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c384-w300-m1200-20261006-r1` |
| 384 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c384-w300-m1200-20261006-r2` |
| 512 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c512-w300-m1200-20261006-r1` |
| 512 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c512-w300-m1200-20261006-r1` |
| 512 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c512-w300-m1200-20261006-r1` |
| 512 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c512-w300-m1200-20261006-r1` |
| 576 | 当前新方法 | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1` |
| 576 | Colocated baseline | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-colocated-c576-w300-m1200-20261006-r1` |
| 576 | No-reverse PD | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c576-w300-m1200-20261006-r1` |
| 576 | 原生 Mooncake | `/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-native-mooncake-c576-w300-m1200-20261006-r1` |

## 运行备注

- 新方法 c384 的首次运行因 1000 GiB 作业 cgroup 触发内存上限而废弃；表中使用 1800 GiB 作业的 r2，有效窗口内无 cgroup 内存事件。
- 新方法 c512/c576 仅出现孤立的前端单次连接重试，均未进入第二次 attempt，最终失败数为 0。
- No-reverse c512 的 analyzer 已完整生成边界、摘要与图；其 wrapper 在测量结束后因运行期间 launcher 文件发生更新而以 2 退出，不影响测量数据有效性。
- 原生 Mooncake 统一使用 256 GB 共享 segment、P HiCache 128 GiB/实例、D HiCache 56 GiB/实例、P write-through、物理 GPU 本地 NUMA 绑定和 1200 秒 scheduler watchdog。
- Mooncake c384 的首次运行因启动环境缺少 `libcudart.so.12` 路径而在模型启动前退出，未产生业务数据；表中使用修复环境后的 r2。
- c64/c128/c256 的 12 组由自动矩阵逐组运行；每组结束后均通过 GPU、端口、Apptainer/FUSE helper 清理门禁。

更完整的逐组测量说明仍保留在 [A100/H100 汇总记录](BROWSECOMP_QWEN3_8B.md)。
