#!/usr/bin/env python3
"""Serial c512 DRAM ablation. Reuse launchers; isolate byte-counter observers."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from dram_resource_sampler import Sampler, summarize as summarize_resources


ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022")
REPORT = ROOT / "runs-host/BROWSECOMP_QWEN3_8B_H100_DRAM_ABLATION.md"
OBSERVER = Path(__file__).with_name("dram_observer")
VARIANTS = {
    "agentic-full": dict(method="new_method", d2p=128, p2d=64, label="新方法512+256 GiB"),
    "agentic-half": dict(method="new_method", d2p=64, p2d=32, label="新方法256+128 GiB"),
    "hicache-full": dict(method="native_mooncake", p=128, d=64, label="HiCache P128/D64 GB +256 GiB"),
    "hicache-reduced": dict(method="native_mooncake", p=96, d=48, label="HiCache P96/D48 GB +256 GiB"),
}
CASES = [(name, spec["method"], 512) for name, spec in VARIANTS.items()]
spec = importlib.util.spec_from_file_location(
    "browsecomp_dram_lowc", Path(__file__).with_name("browsecomp_qwen3_8b_lowc_h100_matrix.py"))
lowc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lowc
spec.loader.exec_module(lowc)
base = lowc.base
base.CONTROL = CONTROL
base.CASES = CASES
original_summarize = base.summarize
samplers = {}


def case_key(case):
    return "dram-4p4d-" + case[0] + "-c512"


def run_dir(case):
    return CONTROL / ("browsecomp8b-" + case[0] + "-c512-w300-m1200-r2")


def variant(case):
    cfg = dict(VARIANTS[case[0]])
    choices = CONTROL / "capacity-choice.json"
    if choices.exists() and case[0] == "hicache-reduced":
        cfg["d"] = int(json.loads(choices.read_text())["hicache_reduced_d_gb"])
        cfg["label"] = f"HiCache P96/D{cfg['d']} GB +256 GiB"
    return cfg


def case_environment(case, directory):
    cfg = variant(case)
    if cfg["method"] == "new_method":
        env, command = lowc.modified_environment(case, directory)
        env.update(D2P_HOST_ARENA_GIB_PER_P=str(cfg["d2p"]),
                   P2D_HOST_ARENA_GIB_PER_P=str(cfg["p2d"]),
                   SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=str(cfg["d2p"]),
                   SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=str(cfg["p2d"]))
        source = CONTROL / "code/new_method"
        env["PYTHONPATH"] = f"{source}:{env['PYTHONPATH']}"
    else:
        env, command = lowc.baseline_environment(case, directory)
        env.update(P_HICACHE_SIZE=str(cfg["p"]), D_HICACHE_SIZE=str(cfg["d"]))
        env["PYTHONPATH"] = str(CONTROL / "code/baseline")
    env.update(PD_DRAM_TRACE_DIR=str(directory / "dram-pools"), PD_DRAM_SAMPLE_SECONDS="2")
    # All inputs are local; avoid persisting unrelated credentials in manifests.
    env.pop("HF_TOKEN", None)
    env["PYTHONPATH"] = f"{OBSERVER}:{env['PYTHONPATH']}"
    return env, command


def validate_inputs():
    lowc.validate_inputs()
    for path in (CONTROL / "code/new_method/sglang/__init__.py",
                 CONTROL / "code/baseline/sglang/__init__.py", OBSERVER / "sitecustomize.py"):
        if not path.exists():
            raise RuntimeError(f"isolated code snapshot missing: {path}")


def cleanup_gate(case):
    env, _ = case_environment(case, run_dir(case))
    base.wait_ports_free(env, timeout=300)
    occupied = base.gpu_processes()
    if occupied:
        raise RuntimeError(f"post-case GPU cleanup gate failed: {occupied}")
    helpers = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if (path / "comm").read_text().strip() in {"apptainer", "squashfuse_ll", "fuse-overlayfs"}:
                helpers.append(int(path.name))
        except OSError:
            pass
    if helpers:
        raise RuntimeError(f"post-case Apptainer/FUSE gate failed: {helpers}; do not launch next case")


def render_report(states):
    lines = ["# BrowseComp Qwen3-8B H100 DRAM消融（c512）", "",
             "单节点node2700，8×H100，TP=1、4P:4D；300秒业务预热＋1200秒正式窗口。",
             "四种配置全部重跑；普通卡mem_fraction_static=0.80、GPU7=0.60。",
             "新方法容量单位GiB；原生HiCache参数单位十进制GB；Mooncake segment为256 GiB。", "",
             "有效KV占用由pool/arena字节计数直接采样，PSS为全部本组进程的去重共享内存占用。",
             "每2秒保留完整Prometheus响应、分池用量、节点内存和cgroup计数。",
             "进程PSS独立线程每60秒扫描一次，记录扫描起止时间；2秒曲线携带最近PSS及其年龄，不把它当成2秒实测。", "",
             "PSS峰值是60秒采样最大值；KV池有效用量峰值是2秒采样最大值，均非瞬时硬件最高水位。",
             "r1因同步PSS扫描拖慢采样而作废并保留原始数据；r2从本组完整300+1200秒重新开始。", "",
             "| 配置 | 状态 | Decode tok/s | Parent KV复用 | 有效KV均值/峰值GiB | 进程PSS均值/峰值GiB | 覆盖率 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for case in CASES:
        detail = states.get(case_key(case), {})
        result = detail.get("result", {})
        resource = result.get("host_resources", {})
        fields = resource.get("fields", {})
        def pair(name):
            row = fields.get(name, {})
            return f"{row['mean_gib']:.2f} / {row['peak_gib']:.2f}" if row.get("mean_gib") is not None else "—"
        throughput = f"{result['decode_tps']:.1f}" if "decode_tps" in result else "—"
        reuse = f"{result['parent_kv_reuse']:.2%}" if "parent_kv_reuse" in result else "—"
        coverage = f"{resource['valid_time_fraction']:.2%}" if resource else "—"
        lines.append(f"| {variant(case)['label']} | {detail.get('status', 'pending')} | {throughput} | {reuse} | "
                     f"{pair('total_kv_used_bytes')} | {pair('process_pss_bytes')} | {coverage} |")
    lines += ["", "## 原始曲线数据", ""]
    for case in CASES:
        directory = run_dir(case)
        lines.append(f"- {variant(case)['label']}：`{directory}`；绘图输入`host_resources_2s.csv`，"
                     "完整原始响应`raw_prometheus_2s.jsonl.gz`，分池计数`dram-pools/*.jsonl`。")
    lines += ["", "48 GB小于历史普通D GPU KV池51.01 GB；若仍使用原生单位，第四组须先确认D容量调整。",
              "容量检查保留原生断言；有效KV占用、固定容量和整节点内存分别记录。", ""]
    REPORT.write_text("\n".join(lines))


def atomic_json(path, value):
    if path == CONTROL / "state.json" and isinstance(value, dict):
        previous = json.loads(path.read_text()) if path.exists() else {}
        for case in CASES:
            key = case_key(case)
            now = value.get(key, {})
            old = previous.get(key, {})
            if now.get("status") == "running" and old.get("status") != "running":
                cfg = variant(case)
                if cfg["method"] == "native_mooncake" and cfg["d"] * 1e9 <= 51007979520:
                    raise RuntimeError("48 decimal GB < 51.01 GB GPU KV pool; awaiting capacity-choice.json")
                directory = run_dir(case)
                env, _ = case_environment(case, directory)
                endpoints = [f"http://127.0.0.1:{p}" for p in
                             (env["PREFILL_PORTS"] + " " + env["DECODE_PORTS"]).split()]
                master = (f"http://127.0.0.1:{env['MOONCAKE_METRICS_PORT']}"
                          if cfg["method"] == "native_mooncake" else None)
                sampler = Sampler(directory, cfg["method"], endpoints, master)
                samplers[str(directory)] = sampler
                sampler.start()
            if now.get("status") == "complete" and old.get("status") != "complete":
                cleanup_gate(case)
            if now.get("status") == "failed":
                sampler = samplers.pop(str(run_dir(case)), None)
                if sampler:
                    sampler.stop()
    lowc.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


def summarize(directory, before_events):
    sampler = samplers.pop(str(directory), None)
    if sampler:
        sampler.stop()
    result = original_summarize(directory, before_events)
    result["host_resources"] = summarize_resources(directory)
    return result


base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.validate_inputs = validate_inputs
base.atomic_json = atomic_json
base.summarize = summarize


if __name__ == "__main__":
    CONTROL.mkdir(parents=True, exist_ok=True)
    state_path = CONTROL / "state.json"
    cleanup_gate(CASES[0])
    lowc.original_atomic_json(CONTROL / "initial-cleanup-gate-r2.json",
                             dict(ts=time.time(), status="clean", gpu_processes=0,
                                  configured_ports_busy=[], apptainer_fuse_helpers=0))
    render_report(json.loads(state_path.read_text()) if state_path.exists() else {})
    manifest = CONTROL / "code_identity.json"
    if not manifest.exists():
        files = {}
        for directory in (CONTROL / "code/new_method", CONTROL / "code/baseline", OBSERVER):
            for path in sorted(directory.rglob("*.py")):
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        lowc.original_atomic_json(manifest, dict(created_at=time.time(), node="node2700", files=files))
    base.main()
