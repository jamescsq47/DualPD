#!/usr/bin/env python3
"""One isolated new-method run: per-P D2P=64 GiB, P2D=128 GiB."""
from __future__ import annotations

# Resolve only maintained core/shared tools; historical siblings stay local.
if __package__:
    from ._core import load_core as _load_core
else:
    from _core import load_core as _load_core


import hashlib
import json
import os
from pathlib import Path
import socket
import time

if __package__:
    from . import browsecomp_qwen3_8b_dram_h100_matrix as dram
else:
    import browsecomp_qwen3_8b_dram_h100_matrix as dram
VIEW = _load_core('dram_complete_frame_analysis').VIEW
prepare_analysis = _load_core('dram_complete_frame_analysis').prepare_analysis

SOURCE_CONTROL = dram.CONTROL
CONTROL = SOURCE_CONTROL.parent / "h100-browsecomp-qwen3-8b-dram-d64p128-c512-20261028"
REPORT = dram.REPORT
FROZEN = SOURCE_CONTROL / "code/new_method"
PRIOR = SOURCE_CONTROL / "browsecomp8b-agentic-full-c512-w300-m1200-r2/launch_config.json"
CASE = ("agentic-d64p128", "new_method", 512)
START = "<!-- DRAM_D64P128_START -->"
END = "<!-- DRAM_D64P128_END -->"


def run_dir(case):
    attempt = int(os.environ.get("PD_DRAM_ATTEMPT", "1"))
    if attempt < 1:
        raise RuntimeError("invalid experiment attempt")
    return CONTROL / f"browsecomp8b-agentic-d64p128-c512-w300-m1200-r{attempt}"


def case_environment(case, directory):
    # Replay the recorded environment, not today's defaults or inherited knobs.
    previous = json.loads(PRIOR.read_text())
    env = dict(previous["env"])
    env.update(RUN_DIR=str(directory), PD_DRAM_TRACE_DIR=str(directory / "dram-pools"),
               D2P_HOST_ARENA_GIB_PER_P="64", P2D_HOST_ARENA_GIB_PER_P="128",
               SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="64",
               SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB="128")
    env.pop("HF_TOKEN", None)
    return env, list(previous["command"])


def validate_inputs():
    dram.lowc.validate_inputs()
    manifest = json.loads((SOURCE_CONTROL / "code_identity.json").read_text())
    checked = 0
    for name, digest in manifest["files"].items():
        path = Path(name)
        if path.is_relative_to(FROZEN):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"original frozen engine changed: {path}")
            checked += 1
    if not checked or not (FROZEN / "sglang/__init__.py").exists():
        raise RuntimeError("original frozen engine snapshot unavailable")


def summarize(directory, before_events):
    sampler = dram.samplers.pop(str(directory), None)
    if sampler:
        sampler.stop()
    result = dram.original_summarize(Path(directory) / VIEW, before_events)
    result["host_resources"] = dram.summarize_resources(directory)
    result["telemetry_validation"] = json.loads((Path(directory) / "complete_frame_validation.json").read_text())
    return result


def render_report(states):
    item = states.get(dram.case_key(CASE), {})
    result = item.get("result", {})
    resources = result.get("host_resources", {})
    fields = resources.get("fields", {})
    def pair(name):
        row = fields.get(name, {})
        return f"{row['mean_gib']:.2f} / {row['peak_gib']:.2f}" if row.get("mean_gib") is not None else "—"
    throughput = f"{result['decode_tps']:.1f}" if "decode_tps" in result else "—"
    reuse = f"{result['parent_kv_reuse']:.2%}" if "parent_kv_reuse" in result else "—"
    coverage = (f"{resources['valid_time_fraction']:.2%} / {resources['process_pss_valid_time_fraction']:.2%}"
                if resources else "—")
    node = os.environ.get("SLURMD_NODENAME", "待分配独占8×H100节点")
    lines = [START, "## 追加：新方法 D→P 256＋P→D 512 GiB（c512）", "",
             f"节点：{node}。Qwen3-8B、TP=1、4P:4D、c512；每P的D→P池64 GiB、P→D池128 GiB。",
             "全节点D→P 256 GiB＋P→D 512 GiB，总计768 GiB。复用原新方法冻结引擎和记录的启动环境，",
             "仅改两方向池容量和本组记录路径；普通GPU显存比例0.80，GPU7为0.60。",
             "完整300秒预热＋1200秒稳态；Host池/节点/cgroup每2秒采样，进程PSS独立每60秒扫描。", "",
             "| 配置 | 状态 | Decode tok/s | Parent KV复用 | KV池已分配量均值/峰值GiB | 整组PSS均值/峰值GiB | KV/PSS覆盖率 |",
             "|---|---|---:|---:|---:|---:|---:|",
             f"| 新方法256+512 GiB | {item.get('status', 'pending')} | {throughput} | {reuse} | "
             f"{pair('total_kv_used_bytes')} | {pair('process_pss_bytes')} | {coverage} |", "",
             f"原始记录：`{run_dir(CASE)}`。曲线：`host_resources_2s.csv`、`host_pools_2s.csv`；"
             "实际PSS扫描：`process_memory_60s.jsonl`。",
             "固定池容量、池内已分配量和整组PSS不是同一指标；PSS是60秒非原子扫描，峰值仅为采样峰值。"]
    if result:
        lines += ["", f"D→P已分配量均值/峰值：{pair('d2p_used_bytes')} GiB；"
                  f"P→D：{pair('p2d_used_bytes')} GiB。",
                  f"实际预热{result['warmup_seconds']:.3f}秒、稳态{result['measurement_seconds']:.3f}秒；"
                  f"请求失败{result['failures']}次；OOM增量见h100_result.json。",
                  "吞吐使用P/D全部八端点完整共同帧；缺失端点不补零，原始metrics保留不变。"]
    if item.get("status") == "complete":
        lines += ["", "complete前已确认GPU进程=0、实验端口连续三次空闲、Apptainer/FUSE辅助进程=0。"]
    lines.append(END)
    text = REPORT.read_text()
    block = "\n".join(lines)
    if START in text:
        before, rest = text.split(START, 1)
        if END not in rest:
            raise RuntimeError("incomplete managed report block; refusing to overwrite")
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    temporary = REPORT.with_suffix(".d64p128.tmp")
    temporary.write_text(text)
    os.replace(temporary, REPORT)


# Only this controller process is patched; engine and existing runs are untouched.
dram.CONTROL = CONTROL
dram.CASES = [CASE]
dram.VARIANTS = {CASE[0]: dict(method="new_method", d2p=64, p2d=128, label="新方法256+512 GiB")}
dram.run_dir = run_dir
dram.case_environment = case_environment
dram.validate_inputs = validate_inputs
dram.render_report = render_report
dram.base.CONTROL = CONTROL
dram.base.CASES = [CASE]
dram.base.run_dir = run_dir
dram.base.case_environment = case_environment
dram.base.validate_inputs = validate_inputs
dram.base.ensure_analysis = prepare_analysis
dram.base.summarize = summarize


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    audit = json.loads((CONTROL / "audit-go.json").read_text())
    if audit.get("verdict") != "GO":
        raise RuntimeError("independent audit GO missing")
    validate_inputs()
    dram.cleanup_gate(CASE)
    def record_gate(name):
        dram.lowc.original_atomic_json(CONTROL / name,
            dict(ts=time.time(), node=socket.gethostname(), status="clean",
                 gpu_processes=0, busy_configured_ports=[], apptainer_fuse_helpers=0))
    record_gate("initial-cleanup-gate.json")
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((FROZEN / "sglang").rglob("*.py"))}
    for p in (Path(__file__), Path(dram.__file__), PRIOR, Path(__file__).with_name("_core.py"),
              dram.ROOT / "scripts/cluster/dram_resource_sampler.py",
              dram.ROOT / "scripts/cluster/dram_complete_frame_analysis.py", dram.OBSERVER / "sitecustomize.py",
              *dram.base.PD.joinpath("scripts/new_method").rglob("*.sh"),
              *dram.base.PD.joinpath("scripts/replay/new_method").rglob("*.sh")):
        files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    dram.lowc.original_atomic_json(CONTROL / "code_identity.json",
        dict(created_at=time.time(), node=socket.gethostname(), frozen_engine=str(FROZEN), files=files))
    dram.base.main()
    dram.cleanup_gate(CASE)
    record_gate("final-cleanup-gate.json")


if __name__ == "__main__":
    main()
