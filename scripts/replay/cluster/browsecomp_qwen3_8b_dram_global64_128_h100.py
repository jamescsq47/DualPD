#!/usr/bin/env python3
"""Requested GLOBAL Host capacities: D2P=64 GiB, P2D=128 GiB, 4P:4D."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

if __package__:
    from . import browsecomp_qwen3_8b_dram_d64p128_h100 as prior
else:
    import browsecomp_qwen3_8b_dram_d64p128_h100 as prior

dram = prior.dram
SOURCE_CONTROL = prior.SOURCE_CONTROL
CONTROL = SOURCE_CONTROL.parent / "h100-browsecomp-qwen3-8b-dram-global64-128-c512-20261028"
REPORT = prior.REPORT
CASE = ("agentic-global64-128", "new_method", 512)
START = "<!-- DRAM_GLOBAL64_128_START -->"
END = "<!-- DRAM_GLOBAL64_128_END -->"
original_environment = prior.case_environment
original_summarize = prior.summarize


def run_dir(case):
    attempt = int(os.environ.get("PD_DRAM_ATTEMPT", "1"))
    if attempt < 1:
        raise RuntimeError("invalid experiment attempt")
    return CONTROL / f"browsecomp8b-agentic-global64-128-c512-w300-m1200-r{attempt}"


def check_global_capacities(env):
    if len(env["PREFILL_GPUS"].split()) != 4 or env["PREFILL_TP_SIZE"] != "1":
        raise RuntimeError("global Host capacity split requires four TP=1 P workers")
    expected = {"D2P_HOST_ARENA_GIB_PER_P": 16, "P2D_HOST_ARENA_GIB_PER_P": 32,
                "SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB": 16,
                "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB": 32}
    for key, value in expected.items():
        if int(env[key]) != value:
            raise RuntimeError(f"GLOBAL 64/128 GiB capacity mismatch: {key}={env[key]}")


def case_environment(case, directory):
    env, command = original_environment(case, directory)
    env.update(D2P_HOST_ARENA_GIB_PER_P="16", P2D_HOST_ARENA_GIB_PER_P="32",
               SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="16",
               SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB="32")
    check_global_capacities(env)
    return env, command


def summarize(directory, before_events):
    result = original_summarize(directory, before_events)
    boundary = json.loads((Path(directory) / "closed_loop_boundaries.json").read_text())
    with (Path(directory) / "host_resources_2s.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not boundary["measurement_start_wall"] <= row["ts"] <= boundary["measurement_end_wall"]:
                continue
            if not row.get("valid_sample"):
                continue
            for direction, capacity in (("d2p", 64), ("p2d", 128)):
                pools = [p for p in row["pools"] if p["kind"] == "arena" and p["direction"] == direction]
                if len(pools) != 4 or sum(p["capacity_bytes"] for p in pools) != capacity * 1024**3:
                    raise RuntimeError(f"observed GLOBAL {direction} capacity differs from {capacity} GiB")
    result["global_host_capacity_gib"] = dict(d2p=64, p2d=128, total=192)
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
    node = os.environ.get("SLURMD_NODENAME", "待调度独占8×H100节点")
    lines = [START, "## 用户指定配置：全节点 D→P 64＋P→D 128 GiB（c512）", "",
             "此前256+512 GiB是将用户的全节点64/128误解为每P容量而产生的额外实验，不是本组结果。",
             f"节点：{node}；Qwen3-8B、TP=1、4P:4D、c512。",
             "**全节点D→P=64 GiB、P→D=128 GiB，总计192 GiB；每P分别16/32 GiB。**",
             "沿用之前Host参数的GiB计量口径；GPU布局P=0/2/4/6、D=1/3/5/7；普通卡0.80，GPU7为0.60。",
             "其余参数、冻结SGLang、数据与采样口径保持原样；300秒预热＋1200秒稳态，Host池/节点/cgroup每2秒，PSS每60秒。", "",
             "| 全节点D→P＋P→D | 每P D→P/P→D | 状态 | Decode tok/s | Parent KV复用 | KV已分配量均值/峰值GiB | 整组PSS均值/峰值GiB |",
             "|---|---|---|---:|---:|---:|---:|",
             f"| 64＋128 GiB | 16/32 GiB | {item.get('status', 'pending')} | {throughput} | {reuse} | "
             f"{pair('total_kv_used_bytes')} | {pair('process_pss_bytes')} |", "",
             f"原始记录：`{run_dir(CASE)}`；曲线`host_resources_2s.csv`、`host_pools_2s.csv`，PSS扫描`process_memory_60s.jsonl`。",
             "固定容量、池内已分配量、整组PSS分开统计；PSS为非原子60秒扫描，峰值为采样峰值。"]
    if result:
        lines += ["", f"D→P已分配量均值/峰值{pair('d2p_used_bytes')} GiB；P→D {pair('p2d_used_bytes')} GiB。",
                  f"预热{result['warmup_seconds']:.3f}秒、稳态{result['measurement_seconds']:.3f}秒；"
                  f"请求失败{result['failures']}次；KV/PSS覆盖率"
                  f"{resources['valid_time_fraction']:.2%}/{resources['process_pss_valid_time_fraction']:.2%}。",
                  "正式窗口逐条核验四个D→P池合计64 GiB、四个P→D池合计128 GiB，不能以每P容量代替总量。"]
    if item.get("status") == "complete":
        lines += ["", "完成前已通过GPU进程=0、配置端口连续三次空闲、Apptainer/FUSE=0门禁。"]
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
    temporary = REPORT.with_suffix(".global64-128.tmp")
    temporary.write_text(text)
    os.replace(temporary, REPORT)


# Isolated process-local controller binding; no shared engine code edits.
prior.CONTROL = CONTROL
prior.CASE = CASE
prior.run_dir = run_dir
prior.case_environment = case_environment
prior.render_report = render_report
dram.CONTROL = CONTROL
dram.CASES = [CASE]
dram.VARIANTS = {CASE[0]: dict(method="new_method", d2p=16, p2d=32, label="全节点64+128 GiB")}
dram.run_dir = run_dir
dram.case_environment = case_environment
dram.render_report = render_report
dram.base.CONTROL = CONTROL
dram.base.CASES = [CASE]
dram.base.run_dir = run_dir
dram.base.case_environment = case_environment
dram.base.summarize = summarize


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    env, _ = case_environment(CASE, run_dir(CASE))
    check_global_capacities(env)
    dram.lowc.original_atomic_json(CONTROL / "requested-global-capacities.json",
        dict(ts=time.time(), global_d2p_gib=64, global_p2d_gib=128, global_total_gib=192,
             per_p_d2p_gib=16, per_p_p2d_gib=32, logical_prefill_workers=4,
             controller=str(Path(__file__)), controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    prior.main()


if __name__ == "__main__":
    main()
