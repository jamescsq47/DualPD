"""Entire-file proof for common configuration and fail-closed validation."""
from pathlib import Path
import json
import re
import subprocess

BASE = "0ed608d2906ed0f3b1b0b27f33e38b9ebda75495"
ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT
PREFIX = "scripts/new_method/"
NAMES = ("run_1p3d_case.sh", "run_1p5d_case.sh", "run_2p6d_numa_case.sh", "run_4p4d_numa_case.sh")
KEYS = ("SGLANG_PD_LATE_BIND_POLL_INTERVAL_S", "SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S",
        "SGLANG_PD_LATE_BIND_QUEUE_TIMEOUT_S", "PD_LATE_BIND_READY_TIMEOUT_S",
        "CLOSED_LOOP_WARMUP_MIN_SECONDS", "CLOSED_LOOP_WARMUP_COMPLETIONS",
        "CLOSED_LOOP_RECENT_SECONDS", "CLOSED_LOOP_MAX_WARMUP_SECONDS",
        "CLOSED_LOOP_MEASUREMENT_SECONDS", "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD",
        "SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "SGLANG_AGENTIC_KV_EARLY_CLAIM",
        "SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT",
        "MC_TCP_ENABLE_CONNECTION_POOL", "SERVICE_STARTUP_TIMEOUT_SECONDS")
PATTERN = re.compile(r'(?<!\S)([A-Z][A-Z0-9_]*)=(?:"[^"\n]*"|[^\s\\]+)')


def original(relative):
    return subprocess.check_output(["git", "-C", str(REPOSITORY), "show", BASE + ":" + relative], text=True)


def verify(root=ROOT):
    sources = {name: original(PREFIX + name) for name in NAMES}
    maps = []
    for name in NAMES:
        body = sources[name].split("\nenv \\\n", 1)[1]
        matches = list(PATTERN.finditer(body))
        bindings = {m.group(1): m.group(0) for m in matches}
        assert len(matches) == len(bindings), (name, "duplicate target")
        maps.append(bindings)
    common = [(key, maps[0][key]) for key in KEYS]
    assert all(all(mapping.get(key) == value for mapping in maps) for key, value in common)
    helper = '''#!/usr/bin/env bash
# Shared case-level settings only; topology and local/TCP policies stay separate.
# Build argv entries, not exported variables. Preserve historical case defaults
# and caller-knob expansion exactly; formal profiles can explicitly select1/1.
dualpd_build_agentic_case_environment() {
  DUALPD_AGENTIC_COMMON_ENV=(
''' + "".join("    " + value + "\n" for key, value in common) + '''  )
}
'''
    assert (root / PREFIX / "internal/agentic_case_environment.sh").read_text() == helper, "complete shared helper changed"
    for name, source in sources.items():
        before, body = source.split("\nenv \\\n", 1)
        for key, value in common:
            assert body.count(value) == 1, (name, key)
            body = body.replace(value, "", 1)
        body = re.sub(r"(?m)^ +\\\n", "", body)
        expected = (before + '\n\nsource "${SCRIPT_DIR}/internal/agentic_case_environment.sh"\n'
                    'dualpd_build_agentic_case_environment\nenv \\\n'
                    '  "${DUALPD_AGENTIC_COMMON_ENV[@]}" \\\n' + body)
        assert (root / PREFIX / name).read_text() == expected, (name, "complete wrapper changed")
    profile_relative = PREFIX + "internal/apply_experiment_profile.sh"
    profile = original(profile_relative)
    profile = profile.replace("  local key value origin\n", "  local key value origin profile_records\n")
    profile = profile.replace('  profile_path="$(realpath -- "${profile_path}")"\n',
                              '  profile_path="$(realpath -- "${profile_path}")" || return\n'
                              '  profile_records="$("${python_bin}" "${loader_path}" "${profile_path}")" || return\n')
    profile = profile.replace('  done < <("${python_bin}" "${loader_path}" "${profile_path}")',
                              '  done <<< "${profile_records}"')
    assert (root / profile_relative).read_text() == profile, "complete profile error-contract change differs"
    loader_relative = "scripts/tools/load_experiment_profile.py"
    assert (root / loader_relative).read_text() == original(loader_relative), "loader/schema/allowlist changed"
    port_result = verify_ports(root)
    return {"base_commit": BASE, "complete_wrappers_verified": 4,
            "samehost_port_validation": port_result,
            "exact_common_bindings": 16, "whole_helper_and_profile_function_verified": True,
            "loader_schema_and_allowlist_unchanged": True,
            "defaults_and_local_tcp_scope_unchanged": True,
            "scope": "Source/configuration proof only; captured argv tests stub fingerprint checks. Native/GPU acceptance is separate."}


def verify_ports(root=ROOT):
    relative = "scripts/cluster/multinode.py"
    expected = original(relative)
    edits = [
        ('    occupied = {router_node[k] for k in worker_ports}\n',
         '    occupied = {n[k] for n in nodes if n["host_ip"] == router_node["host_ip"]\n                for k in worker_ports}\n'),
        ('    local_ports = {relay_node[k] for k in worker_ports}\n',
         '    local_ports = {n[k] for n in nodes if n["host_ip"] == relay_node["host_ip"]\n                   for k in worker_ports}\n'),
        ('    if control["node_id"] == router["node_id"]:\n        local_ports.update',
         '    if relay_node["host_ip"] == router_node["host_ip"]:\n        local_ports.update'),
    ]
    for before, after in edits:
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    assert (root / relative).read_text() == expected, "complete TCP launcher differs beyond three port validations"
    return {"complete_module_verified": True, "exact_validation_edits": 3,
            "scope": "Union by explicit host_ip only; no DNS/multi-NIC identity inference. Valid configuration defaults/commands unchanged; colliding same-IP listeners now fail closed."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
