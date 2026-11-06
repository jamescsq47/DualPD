#!/usr/bin/env python3
"""One native HiCache P64/D64 follow-up; preserve the completed four cases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time

import browsecomp_qwen3_8b_dram_h100_matrix as dram
from dram_complete_frame_analysis import VIEW, prepare_analysis


SOURCE_CONTROL = dram.CONTROL
CONTROL = SOURCE_CONTROL.parent / "h100-browsecomp-qwen3-8b-dram-p64d64-c512-20261022"
REPORT = dram.REPORT
BASELINE = SOURCE_CONTROL / "code/baseline"
CASE = ("hicache-p64d64", "native_mooncake", 512)
START = "<!-- DRAM_P64D64_START -->"
END = "<!-- DRAM_P64D64_END -->"


def run_dir(case):
    return CONTROL / "browsecomp8b-hicache-p64d64-c512-w300-m1200-r1"


def case_environment(case, directory):
    env, command = dram.lowc.baseline_environment(case, directory)
    env.update(P_HICACHE_SIZE="64", D_HICACHE_SIZE="64",
               PD_DRAM_TRACE_DIR=str(directory / "dram-pools"), PD_DRAM_SAMPLE_SECONDS="2",
               PYTHONPATH=f"{dram.OBSERVER}:{BASELINE}")
    env.pop("HF_TOKEN", None)
    return env, command


def validate_inputs():
    dram.lowc.validate_inputs()
    if not (BASELINE / "sglang/__init__.py").exists():
        raise RuntimeError(f"original frozen baseline missing: {BASELINE}")
    if not (dram.OBSERVER / "sitecustomize.py").exists():
        raise RuntimeError("original DRAM observer missing")


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
    node = os.environ.get("SLURMD_NODENAME", "待调度：任一独占8×H100、约2TB DRAM节点")
    lines = [START, "## 追加：HiCache P64/D64（c512）", "",
             f"节点：{node}。TP=1、4P:4D；P64 GB×4+D64 GB×4+Mooncake256 GiB。",
             "复用上述原生baseline冻结快照、数据、GPU布局及显存比例，仅将P池改成64 GB；不覆盖原四组。",
             "300秒业务预热＋1200秒正式窗口；KV/节点/cgroup每2秒，PSS独立每60秒采样。", "",
             "| 配置 | 状态 | Decode tok/s | Parent KV复用 | KV池占用均值/峰值GiB | 进程PSS均值/峰值GiB | KV / PSS覆盖率 |",
             "|---|---|---:|---:|---:|---:|---:|",
             f"| HiCache P64/D64 GB +256 GiB | {item.get('status', 'pending')} | {throughput} | {reuse} | "
             f"{pair('total_kv_used_bytes')} | {pair('process_pss_bytes')} | {coverage} |", "",
             f"原始记录：`{run_dir(CASE)}`。绘图输入`host_resources_2s.csv`、`host_pools_2s.csv`；"
             "PSS实际扫描见`process_memory_60s.jsonl`；完整指标见`raw_prometheus_2s.jsonl.gz`。"]
    if result:
        lines += ["", f"实测预热{result['warmup_seconds']:.3f}秒、正式{result['measurement_seconds']:.3f}秒；"
                  f"正式请求失败{result['failures']}次。",
                  f"P池均值/峰值{pair('prefill_used_bytes')} GiB；D池{pair('decode_used_bytes')} GiB；"
                  f"Mooncake{pair('mooncake_used_bytes')} GiB。"]
        if result.get("telemetry_validation"):
            check = result["telemetry_validation"]
            lines += ["", "统计使用P/D八个端点均完整的共同采样帧，缺失端点不补零；原始metrics未改写。",
                      f"累计计数覆盖{check['counter_span_seconds']:.3f}秒，末120秒完整采样间Decode增加"
                      f"{check['tail_120s_decode_tokens']:.0f} tokens；旧错误聚合结果另存legacy-aggregate文件。",
                      f"引擎完整帧是不规则采样，最大间隔{check['maximum_complete_frame_gap_seconds']:.3f}秒；"
                      "派生引擎吞吐曲线不视为2秒实测，Host内存曲线仍为真实2秒采样。"]
    if item.get("status") == "complete":
        lines += ["", "complete仅在本组GPU=0、实验端口连续三次空闲、Apptainer/FUSE=0后写入。"]
    lines += [END]
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
    temporary = REPORT.with_suffix(".p64d64.tmp")
    temporary.write_text(text)
    os.replace(temporary, REPORT)


# Only this process's controller globals change. No engine package is edited.
dram.CONTROL = CONTROL
dram.CASES = [CASE]
dram.VARIANTS = {CASE[0]: dict(method="native_mooncake", p=64, d=64,
                              label="HiCache P64/D64 GB +256 GiB")}
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
    validate_inputs()
    dram.cleanup_gate(CASE)
    dram.lowc.original_atomic_json(CONTROL / "initial-cleanup-gate.json",
                                  dict(ts=time.time(), node=socket.gethostname(), status="clean",
                                       gpu_processes=0, busy_configured_ports=[], apptainer_fuse_helpers=0))
    # Hash the immutable baseline and read-only samplers used for this case.
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((BASELINE / "sglang").rglob("*.py"))}
    for p in (Path(__file__), Path(dram.__file__),
              Path(__file__).with_name("dram_resource_sampler.py"),
              Path(__file__).with_name("dram_complete_frame_analysis.py"), dram.OBSERVER / "sitecustomize.py"):
        files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    dram.lowc.original_atomic_json(CONTROL / "code_identity.json",
                                  dict(created_at=time.time(), node=socket.gethostname(),
                                       frozen_baseline=str(BASELINE), files=files))
    dram.base.main()
    # Repeat the hard gate and retain an explicit successful final probe.
    dram.cleanup_gate(CASE)
    dram.lowc.original_atomic_json(CONTROL / "final-cleanup-gate.json",
                                  dict(ts=time.time(), node=socket.gethostname(), status="clean",
                                       gpu_processes=0, busy_configured_ports=[], apptainer_fuse_helpers=0))


if __name__ == "__main__":
    main()
