# DualPD 远端执行入口

开发在本地 `dualpd` 工作区进行。用户在目标机器执行 Bash 脚本，将原始报告回传；不要求远端安装 Codex。

以下命令均从独立项目 `DualPD/` 根目录执行，不再使用旧 Slime 的
`tools/dualpd` 路径。

## 当前维护入口与历史回放

当前维护入口包括 `multinode.py`/`multinode.sh`、`qwen35_slurm.py`、
`qwen35_native_pd_slurm.py`、`qwen35_colocated_slurm.py`、
`process_supervisor.py`，以及 `engine_cleanup_*` 验证和 allocation 工具。
BrowseComp 整理回归使用 `engine_cleanup_quickcheck.py`，固定配方及测量组件
位于 `../benchmarks/`，不再导入历史 matrix controller。

模型专用 matrix/autopilot/watchdog 和已记录实验启动脚本已搬到
`../replay/cluster/`，已通过完整来源重建、分层调用合约及独立吞吐回归；
这不等于实跑全部历史作业，也没有删除历史功能。其硬编码路径、
容量、超时和消融开关可能定义历史回放身份，不能直接视为当前默认配置。
归档的 `multinode_smoke.py` 是旧文件控制协议
的探针，不是当前 V2 TCP 模型验收入口。迁移进度和性能门禁见
`../../docs/SEQUENTIAL_ENGINE_CLEANUP.md`；当前完整 CPU 门禁是
`python scripts/check_cpu.py`，不能用默认 pytest 选择范围替代。

## 离线工具的项目地址与站点资源

`dram_complete_frame_analysis.prepare_analysis(directory, *, python=None,
project_root=None)` 默认从当前工具文件定位项目里的
`scripts/tools/analyze_pd_offload.py`，不回到旧工作区找代码。为保留历史分析
依赖，默认解释器仍为 `DEFAULT_ANALYSIS_PYTHON` 指向的站点 `pd` 环境；
在其他环境中应显式传 `python=sys.executable`，也可显式选择 `project_root`。
这两个参数不修改原始指标、窗口或分析算法，显式根目录拼写也不会被 realpath 改写。

`run_swe_sif_shards_in_step.sh` 仍只在 Slurm task 内运行，converter 和
policy 从脚本所在项目定位。默认日志放项目父目录（保留当前站点的 workspace
布局），默认数据、OCI、SIF 与临时目录仍为历史站点配置。迁移机器时应显式设置：

- `SWE_SIF_DATASET`、`SWE_SIF_OCI_LAYOUT`：已准备的数据与 OCI 缓存；
- `SWE_SIF_DIR`、`SWE_SIF_TEMP_BASE`、`SWE_SIF_LOG_DIR`：输出、临时与日志目录；
- `SWE_SIF_PYTHON`：转换解释器，默认仍为 PATH 中的 `python`；
- `SWE_SIF_POLICY`：已批准的 containers/image 签名 policy 文件；
- `SWE_SIF_WORKERS_PER_NODE`：每个 task 的 worker 数，默认4。

项目没有附带放宽验证的默认 policy。若不显式设置 `SWE_SIF_POLICY`，会检查脚本
旁的历史名字 `skopeo-public-policy.json`；所选文件缺失或不可读时，在创建输出目录
或启动 worker 前拒绝执行，不自动生成 policy。应由操作者提供其站点批准的策略。

分片公式、转换参数、失败汇总和 CPU-only 范围均未改变。CPU 合约测试只在
临时目录截获 argv，不执行 Docker/Apptainer/SIF 转换，也不代表转换已成功。

## 硬件和拓扑采集

在每个计划参与实验的节点运行一次。最好在实际运行推理服务的容器和 Python 环境内执行；如怀疑容器隐藏了 RDMA/NUMA 信息，可在宿主机再采一份并注明区别。

```bash
bash scripts/cluster/collect_hardware_topology.sh
```

可选参数示例（IP 请替换为另一节点的地址）：

```bash
bash scripts/cluster/collect_hardware_topology.sh \
  --output /tmp/dualpd-node-a-report \
  --peer 10.0.0.2 \
  --python /path/to/conda/env/bin/python
```

输出目录必须尚不存在，防止覆盖旧结果。默认创建 `/tmp/dualpd-hardware-时间-随机后缀`。脚本最后输出报告和压缩包的绝对路径，将每个节点的 `.tar.gz` 回传即可。

采集内容：

- GPU 型号、容量、即时占用、驱动、MIG、NVLink 状态及 P2P 拓扑。
- CPU/NUMA、CPU 亲和性、内存/cgroup/锁页限制。
- GPU 和网卡的 PCIe/NUMA 归属、PCIe 当前/最大链路速率与宽度。
- NIC 驱动、MTU、IP、路由；RDMA 端口、InfiniBand/RoCE、GID 与 netdev 对应关系。
- UCX/OFED/CUDA 编译器版本及指定 Python 环境的相关包版本/源码位置。
- 每个命令的输出、退出码以及缺失/不支持/超时记录。

脚本不使用 sudo、不安装依赖、不改配置、不启动模型、不跑带宽测试、不修改或终止已有服务。需要 Bash、Linux、GNU timeout、tar；其他工具缺失会记录并继续。每条探测默认最多20秒，超时后终止的只是本脚本启动的探测命令。不会扫描 NFS 数据目录。

`--peer` 只查询本地路由，不验证网络连通性。速率字段是链路信息，不是实测吞吐；硬件清单也不能单独证明 GPUDirect RDMA 已可用。

报告包含主机名、内网 IP、GPU UUID 等拓扑标识，分享前可检查；不采集完整环境变量、密钥、访问令牌或任意进程命令行。
