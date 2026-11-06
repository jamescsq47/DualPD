# MiniMax-M2.7 · 三节点 TP8/EP8 · 1P:2D

状态：**9/9 组均已达到 T450 并完成清理**。不是每组均跑完500题；控制器的 `progress=500` 是切组完成哨兵，不是实际完成题数。

P+Router+verifier=node1800；D0=node2701；D1=node2800。24×H100；不跨节点模型并行。
沿用 MiniMax r6 引擎/配置，c256/384/448；SWE500 source-order，temperature=.6/top_p=.95/top_k=20，
context=131072，chunk/batch=8192，单轮8192 token/最多64轮，mem_fraction_static=.8。
Agentic 工具/Direct各1s，拥堵重算关闭；D→P每D节点512 GiB（合计1024 GiB），P→D总256 GiB。
HiCache每rank64 GiB，Mooncake共享store总512 GiB，wait_complete；原生baseline代码不修改。
Agentic沿用r6 compact_handoff_payload=true、profile_host_prepare=true和P→D/Forward串行开关。
指标窗口均为0–T450；T450按第450道COMPLETED计算，FAILED不充数。达到指标后保存快照、停止并切组。
停止前在途Host/传输由t450-logs保留；强制停止不是自然排空或历史queued=durable=release守恒证明。
这是固定500题T450工程对比，不宣称满足300s预热+1200s稳态验收。

| 顺序 | 并发 | 方法 | T450(s) | Decode tok/s | 完成/失败(T450前) |
|---:|---:|---|---:|---:|---:|
| 1 | 256 | Agentic-PD r6 | 1393.55 | 1485.5 | 450/1 |
| 2 | 256 | HiCache/Mooncake | 1485.73 | 1342.2 | 450/0 |
| 3 | 384 | Agentic-PD r6 | 1460.67 | 1475.5 | 450/0 |
| 4 | 384 | HiCache/Mooncake | 1555.76 | 1424.7 | 450/0 |
| 5 | 448 | Agentic-PD r6 | 1453.51 | 1532.7 | 450/0 |
| 6 | 448 | HiCache/Mooncake | 1605.34 | 1373.4 | 450/1 |
| 7 | 256 | No-reverse PD | 2302.66 | 864.8 | 450/1 |
| 8 | 384 | No-reverse PD | 2459.83 | 839.2 | 450/0 |
| 9 | 448 | No-reverse PD | 2536.74 | 834.2 | 450/1 |

| 并发 | 方法 | P Forward/GPU | D Forward/GPU | P累计Forward(s/GPU) | 实际Prefill(M) | 复用率 | P KV | D KV均值 | D running/组 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | Agentic-PD r6 | 60.40% | 98.11% | 841.7 | 6.432 | 93.44% | 16.71% | 14.46% | 10.32 |
| 256 | HiCache/Mooncake | 51.05% | 83.10% | 758.5 | 6.838 | 93.11% | 20.25% | 21.49% | 9.57 |
| 384 | Agentic-PD r6 | 59.25% | 97.77% | 865.5 | 6.529 | 93.73% | 16.77% | 14.08% | 10.10 |
| 384 | HiCache/Mooncake | 65.63% | 90.41% | 1021.1 | 23.722 | 78.17% | 19.35% | 44.51% | 10.10 |
| 448 | Agentic-PD r6 | 59.39% | 97.82% | 863.2 | 6.734 | 93.69% | 17.71% | 15.53% | 10.71 |
| 448 | HiCache/Mooncake | 67.20% | 93.19% | 1078.8 | 26.734 | 75.78% | 14.50% | 45.69% | 9.46 |
| 256 | No-reverse PD | 96.79% | 95.48% | 2228.9 | 91.652 | 0.00% | 1.94% | 73.26% | 4.89 |
| 384 | No-reverse PD | 96.76% | 94.75% | 2380.1 | 97.635 | 0.00% | 1.92% | 77.80% | 4.76 |
| 448 | No-reverse PD | 96.87% | 94.49% | 2457.4 | 100.712 | 0.00% | 1.92% | 77.98% | 4.79 |

## 完成审计

- Agentic-PD 相对同并发 HiCache 的 T450 分别缩短 **6.20%、6.11%、9.46%**。
- c256 Agentic 的累计 P Forward 仍高于 HiCache（841.7 vs 758.5 s/GPU）；c384/c448 则低于 HiCache。不能笼统宣称所有并发均节省 P Forward。
- T450 前，c256 Agentic、c448 HiCache、c256/c448 No-reverse 各有1个 `verifier_infrastructure_error`；这些 FAILED 未计入450个完成样本。其余五组窗口内无 FAILED。
- c448 HiCache 在T450后另有1个 HTTP流读取异常：`Separator is not found, and chunk exceed the limit`。不计入T450前失败数，不把本矩阵表述为零错误验收。
- 停止前实际 COMPLETED 数按运行顺序为 **483、458、458、493、456、494、465、486、477**。每组均保留 `t450-result.json`、`t450-progress.json` 以及三个worker等组件的 `t450-logs/`。
- 2026-10-18 完成后复核：node1800、node2701、node2800 均无GPU计算进程；仅停止本矩阵服务，保留Slurm配额。采集日志未发现 CUDA error/OOM、AssertionError、Fatal Python error 或 Segmentation fault；这不等于所有应用/工具请求均无异常。

控制与结果目录：`/orcd/scratch/orcd/003/shangy/dualpd/runs/minimax-m27-three-node-1p2d-c256c384c448-20261018-r1/`。
