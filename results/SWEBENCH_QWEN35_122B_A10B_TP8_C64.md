# Qwen3.5-122B-A10B TP8 双节点 SWE500 c64 四方法对比

本实验严格使用 `node1700 + node1701` 两个 H100 节点、SWE-bench Verified
500 题 source-order，并发为 64。模型、采样参数、context、Prefill chunk/batch、
Verifier 与既有 c128/c192/c256 正式矩阵保持一致。

四种方法依次为 Agentic-PD、Colocated、原生单向 PD（无 D→P 回传）和原生
HiCache/Mooncake PD。控制器会等待每组 500 题完整结束、验证结果、停止该组全部
进程并确认两台节点 GPU 清空后，才会启动下一组。

<!-- QWEN35_C64_AUTOPILOT_START -->
## 自动运行进度

> 最后检查：2026-10-21 20:50:59 UTC；状态：`complete`；完整结果 **4/4**。
> 四组 c64 实验均已完成。

| 方法 | 完成/失败 | Resolved | T450 | T450 前 Decode tok/s | T500 | 全程 Prefill tok/s | 全程 Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agentic-PD | 499/1 | 321/500 | 4,025.51 s | 1,344.08 | 6,183.95 s | 2,833.16 | 914.63 |
| Colocated | 500/0 | 317/500 | 3,815.42 s | 1,423.89 | 4,603.94 s | 3,221.12 | 1,241.97 |
| HiCache/Mooncake PD | 500/0 | 311/500 | 5,052.70 s | 1,096.78 | 6,029.08 s | 2,582.61 | 984.08 |
| 原生单向 PD | 500/0 | 315/500 | 10,894.26 s | 500.39 | 11,629.89 s | 44,659.95 | 497.55 |

结果目录：

- `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-122b-agentic-swe500-c64-20261020-r1`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-122b-colocated-swe500-c64-20261020-r3`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-122b-hicache-mooncake-swe500-c64-20261020-r1`
- `/orcd/scratch/orcd/003/shangy/dualpd/runs/q35-122b-native-oneway-swe500-c64-20261020-r2`
<!-- QWEN35_C64_AUTOPILOT_END -->
