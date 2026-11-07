#!/usr/bin/env bash
set -euo pipefail

# c576 keeps hundreds of inbound and outbound HTTP connections live at once.
# Raise the limit for the complete process tree (harness, proxy, search and
# routers), rather than only inside the affinity proxy.
ulimit -n 65536

MODE="${1:-local}"
RUN_ID="${2:-ppd45423-browsecomp-${MODE}-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP_SECONDS="${WARMUP_SECONDS:-90}"
MEASURE_SECONDS="${MEASURE_SECONDS:-300}"
CONCURRENCY="${CONCURRENCY:-8}"
PAIR_COUNT="${PAIR_COUNT:-1}"
LAYOUT="${LAYOUT:-paired}"
TURN_CAP="${TURN_CAP:-1024}"

[[ "${MODE}" == "local" || "${MODE}" == "ordinary" ]] || {
  echo "mode must be local or ordinary" >&2
  exit 2
}
if [[ "${LAYOUT}" == "1p7d" ]]; then
  P_COUNT=1
  D_COUNT=7
elif [[ "${LAYOUT}" == "2p6d" ]]; then
  P_COUNT=2
  D_COUNT=6
elif [[ "${LAYOUT}" == "paired" ]] && \
    [[ "${PAIR_COUNT}" == "1" || "${PAIR_COUNT}" == "4" ]]; then
  P_COUNT="${PAIR_COUNT}"
  D_COUNT="${PAIR_COUNT}"
else
  echo "LAYOUT must be paired (PAIR_COUNT 1 or 4), 1p7d, or 2p6d" >&2
  exit 2
fi

ROOT=/home/shangy/dualpd
PD=${ROOT}/slime/examples/pd
TOOLS=${ROOT}/slime/scripts/cluster
PR_ROOT=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423
SOURCE=${PR_ROOT}/sglang
RUN_DIR=${PR_ROOT}/runs/${RUN_ID}
IMAGE=/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423.sif
MODEL=/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B
APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
PYTHON=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python
HF_HOME=/orcd/scratch/orcd/003/shangy/dualpd/datasets/huggingface-cache
QA_DATA=/orcd/scratch/orcd/003/shangy/dualpd/datasets/browsecomp/bc_train.jsonl
SCHEDULE=${PD}/configs/workloads/fixed_browsecomp_source_order_n680.json
EMBEDDINGS=${PD}/data/browsecomp/artifacts/search/corpus_embeddings.pkl

PROXY_PORT=29000
SEARCH_PORT=28900
if [[ "${LAYOUT}" == "1p7d" ]]; then
  # Keep the sole P on GPU0.  All seven fixed D routers share that P; the
  # search service continues to share GPU7, so D6 retains the 0.60 memory cap.
  P_GPUS=(0)
  D_GPUS=(1 2 3 4 5 6 7)
  P_PORTS=(29200)
  D_PORTS=(29300 29301 29302 29303 29304 29305 29306)
  ROUTER_PORTS=(29100 29101 29102 29103 29104 29105 29106)
  P_BOOTSTRAP_PORTS=(29500)
  D_BOOTSTRAP_PORTS=(29500 29500 29500 29500 29500 29500 29500)
  P_NCCL_PORTS=(29600)
  D_NCCL_PORTS=(29700 29701 29702 29703 29704 29705 29706)
  ROUTER_P_INDEX=(0 0 0 0 0 0 0)
elif [[ "${LAYOUT}" == "2p6d" ]]; then
  # One P per socket/NVSwitch locality group; each P feeds three fixed D
  # routers. A session is still pinned to exactly one D by the affinity proxy.
  P_GPUS=(0 4)
  D_GPUS=(1 2 3 5 6 7)
  P_PORTS=(29200 29201)
  D_PORTS=(29300 29301 29302 29303 29304 29305)
  ROUTER_PORTS=(29100 29101 29102 29103 29104 29105)
  P_BOOTSTRAP_PORTS=(29500 29501)
  D_BOOTSTRAP_PORTS=(29500 29500 29500 29501 29501 29501)
  P_NCCL_PORTS=(29600 29601)
  D_NCCL_PORTS=(29700 29701 29702 29703 29704 29705)
  ROUTER_P_INDEX=(0 0 0 1 1 1)
else
  P_GPUS=(0 2 4 6)
  D_GPUS=(1 3 5 7)
  P_PORTS=(29200 29201 29202 29203)
  D_PORTS=(29300 29301 29302 29303)
  ROUTER_PORTS=(29100 29101 29102 29103)
  P_BOOTSTRAP_PORTS=(29500 29501 29502 29503)
  D_BOOTSTRAP_PORTS=(29500 29501 29502 29503)
  P_NCCL_PORTS=(29600 29601 29602 29603)
  D_NCCL_PORTS=(29700 29701 29702 29703)
  ROUTER_P_INDEX=(0 1 2 3)
fi

mkdir -p "${RUN_DIR}/logs"
printf '%s\n' "${RUN_ID}" >"${RUN_DIR}/run-id"
printf '%s\n' "${MODE}" >"${RUN_DIR}/mode"
printf '%s\n' "$$" >"${RUN_DIR}/controller.pid"

pids=()
cleanup() {
  local pid
  set +e
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 8
  for pid in "${pids[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

write_state() {
  local stage="$1"
  STAGE="${stage}" RUN_ID_VALUE="${RUN_ID}" MODE_VALUE="${MODE}" \
    PAIR_COUNT="${PAIR_COUNT}" P_COUNT="${P_COUNT}" D_COUNT="${D_COUNT}" \
    LAYOUT_VALUE="${LAYOUT}" CONCURRENCY="${CONCURRENCY}" \
    WARMUP_SECONDS="${WARMUP_SECONDS}" MEASURE_SECONDS="${MEASURE_SECONDS}" \
    TURN_CAP="${TURN_CAP}" \
    RUN_DIR_VALUE="${RUN_DIR}" "${PYTHON}" - <<'PY'
import json, os, time
from pathlib import Path
path = Path(os.environ["RUN_DIR_VALUE"]) / "state.json"
value = {
    "epoch": time.time(), "run_id": os.environ["RUN_ID_VALUE"],
    "mode": os.environ["MODE_VALUE"], "stage": os.environ["STAGE"],
    "pair_count": int(os.environ.get("PAIR_COUNT", "1")),
    "p_count": int(os.environ.get("P_COUNT", "1")),
    "d_count": int(os.environ.get("D_COUNT", "1")),
    "layout": os.environ.get("LAYOUT_VALUE", "paired"),
    "concurrency": int(os.environ.get("CONCURRENCY", "8")),
    "warmup_seconds": int(os.environ.get("WARMUP_SECONDS", "90")),
    "measure_seconds": int(os.environ.get("MEASURE_SECONDS", "300")),
    "turn_cap": int(os.environ.get("TURN_CAP", "1024")),
}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
}

wait_http() {
  local label="$1" url="$2" pid="$3" timeout="$4" start
  start="$(date +%s)"
  while true; do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${label} exited before readiness" >&2
      return 1
    fi
    if (( $(date +%s) - start >= timeout )); then
      echo "${label} readiness timed out after ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

ports_to_gate=("${PROXY_PORT}" "${SEARCH_PORT}")
for ((index=0; index<P_COUNT; index++)); do
  ports_to_gate+=("${P_PORTS[index]}" "${P_BOOTSTRAP_PORTS[index]}" "${P_NCCL_PORTS[index]}")
done
for ((index=0; index<D_COUNT; index++)); do
  ports_to_gate+=("${D_PORTS[index]}" "${ROUTER_PORTS[index]}" \
    "${D_BOOTSTRAP_PORTS[index]}" "${D_NCCL_PORTS[index]}")
done
for port in "${ports_to_gate[@]}"; do
  if ss -ltn "sport = :${port}" | tail -n +2 | grep -q .; then
    echo "port ${port} is already occupied" >&2
    exit 1
  fi
done

common_container=(
  "${APPTAINER}" exec --nv --cleanenv
  --env "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  --env "PYTHONPATH=${SOURCE}/python"
  --env "SGLANG_CACHE_DIR=/orcd/pool/004/shangy/dualpd/ppd45423-cache"
  --env "SGLANG_RUST_BUILD_MODE=auto"
  --env "RUSTUP_HOME=/root/.rustup"
  --env "CARGO_HOME=/root/.cargo"
  --env "RUSTUP_TOOLCHAIN=stable"
  --env "MC_TCP_ENABLE_CONNECTION_POOL=true"
  --bind /orcd:/orcd,/home/shangy/dualpd:/home/shangy/dualpd
  "${IMAGE}" /opt/sglang/bin/python
)

write_state starting_models
p_pids=()
d_pids=()
for ((index=0; index<P_COUNT; index++)); do
  setsid "${common_container[@]}" -m sglang.launch_server \
    --model-path "${MODEL}" --host 127.0.0.1 --port "${P_PORTS[index]}" \
    --context-length 40960 --mem-fraction-static 0.80 --skip-server-warmup \
    --enable-metrics --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "${P_BOOTSTRAP_PORTS[index]}" \
    --nccl-port "${P_NCCL_PORTS[index]}" --base-gpu-id "${P_GPUS[index]}" \
    >"${RUN_DIR}/logs/prefill-${index}.log" 2>&1 &
  pids+=("$!")
  p_pids+=("$!")
done

for ((index=0; index<D_COUNT; index++)); do
  d_mem=0.80
  [[ "${D_GPUS[index]}" == "7" ]] && d_mem=0.60
  setsid "${common_container[@]}" -m sglang.launch_server \
    --model-path "${MODEL}" --host 127.0.0.1 --port "${D_PORTS[index]}" \
    --context-length 40960 --mem-fraction-static "${d_mem}" --skip-server-warmup \
    --enable-metrics --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "${D_BOOTSTRAP_PORTS[index]}" \
    --nccl-port "${D_NCCL_PORTS[index]}" --base-gpu-id "${D_GPUS[index]}" \
    --disaggregation-decode-enable-radix-cache \
    --disaggregation-decode-enable-local-prefill \
    --chunked-prefill-size 1024 --enable-mixed-chunk \
    >"${RUN_DIR}/logs/decode-${index}.log" 2>&1 &
  pids+=("$!")
  d_pids+=("$!")
done

for ((index=0; index<P_COUNT; index++)); do
  wait_http "prefill-${index}" "http://127.0.0.1:${P_PORTS[index]}/health" "${p_pids[index]}" 1200
done
for ((index=0; index<D_COUNT; index++)); do
  wait_http "decode-${index}" "http://127.0.0.1:${D_PORTS[index]}/health" "${d_pids[index]}" 1200
done

write_state starting_services
router_pids=()
for ((index=0; index<D_COUNT; index++)); do
  p_index="${ROUTER_P_INDEX[index]}"
  setsid "${APPTAINER}" exec --cleanenv \
    --env "PYTHONPATH=${SOURCE}/python" \
    --env "SGLANG_CACHE_DIR=/orcd/pool/004/shangy/dualpd/ppd45423-cache" \
    --env "SGLANG_RUST_BUILD_MODE=auto" \
    --bind /orcd:/orcd,/home/shangy/dualpd:/home/shangy/dualpd \
    "${IMAGE}" /opt/sglang/bin/python -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:${P_PORTS[p_index]}" \
    --decode "http://127.0.0.1:${D_PORTS[index]}" \
    --host 127.0.0.1 --port "${ROUTER_PORTS[index]}" \
    >"${RUN_DIR}/logs/router-${index}.log" 2>&1 &
  router_pids+=("$!")
  pids+=("$!")
done
for ((index=0; index<D_COUNT; index++)); do
  wait_http "router-${index}" "http://127.0.0.1:${ROUTER_PORTS[index]}/health" "${router_pids[index]}" 300
done

setsid env CUDA_VISIBLE_DEVICES=7 SEARCH_SERVER_GPU_IDS=0 \
  HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_HOME}/datasets" \
  SEARCH_SERVER_EMBEDDING_CACHE="${EMBEDDINGS}" PYTHONPATH="${PD}:${ROOT}/slime" \
  "${PYTHON}" -m dualpd.data.browsecomp.search_server --host 127.0.0.1 --port "${SEARCH_PORT}" \
  >"${RUN_DIR}/logs/search.log" 2>&1 &
pids+=("$!")
search_pid="$!"
wait_http search "http://127.0.0.1:${SEARCH_PORT}/health" "${search_pid}" 1200

pd_endpoints=""
decode_endpoints=""
prefill_ports_csv=""
decode_ports_csv=""
for ((index=0; index<P_COUNT; index++)); do
  [[ -n "${prefill_ports_csv}" ]] && prefill_ports_csv+=,
  prefill_ports_csv+="${P_PORTS[index]}"
done
for ((index=0; index<D_COUNT; index++)); do
  [[ -n "${pd_endpoints}" ]] && pd_endpoints+=, decode_endpoints+=, decode_ports_csv+=,
  pd_endpoints+="http://127.0.0.1:${ROUTER_PORTS[index]}"
  decode_endpoints+="http://127.0.0.1:${D_PORTS[index]}"
  decode_ports_csv+="${D_PORTS[index]}"
done

setsid "${PYTHON}" "${TOOLS}/ppd45423_affinity_proxy.py" \
  --port "${PROXY_PORT}" --mode "${MODE}" \
  --pd-endpoints "${pd_endpoints}" \
  --decode-endpoints "${decode_endpoints}" \
  --event-log "${RUN_DIR}/affinity-events.jsonl" \
  >"${RUN_DIR}/logs/proxy.log" 2>&1 &
pids+=("$!")
proxy_pid="$!"
wait_http proxy "http://127.0.0.1:${PROXY_PORT}/health" "${proxy_pid}" 60

write_state running_workload
cd "${PD}"
env CUDA_VISIBLE_DEVICES= LOCAL_SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}" \
  HF_HOME="${HF_HOME}" PYTHONPATH="${PD}:${ROOT}/slime" \
  BROWSECOMP_TURN_MAX_NEW_TOKENS="${TURN_CAP}" PD_MODEL_HTTP_TRANSPORT=aiohttp \
  "${PYTHON}" inference.py \
  --model "${MODEL}" --qa-data "${QA_DATA}" --math-ratio 0 \
  --router-port "${PROXY_PORT}" --router-request-timeout-seconds 3600 \
  --prefill-port "${P_PORTS[0]}" --prefill-ports "${prefill_ports_csv}" \
  --decode-port "${D_PORTS[0]}" --decode-ports "${decode_ports_csv}" \
  --requests 680 --warmup-requests 0 --dispatch-policy fixed \
  --schedule-file "${SCHEDULE}" --preserve-source-order \
  --request-rate 100 --arrival-distribution fixed --max-inflight "${CONCURRENCY}" \
  --metrics-interval 2 --seed 2026 --temperature 0 --top-p 1 --top-k -1 \
  --max-context-length 40960 --max-response-length 36864 \
  --closed-loop --closed-loop-warmup-min-seconds "${WARMUP_SECONDS}" \
  --closed-loop-warmup-completions 0 --closed-loop-recent-seconds 60 \
  --closed-loop-max-warmup-seconds "$((WARMUP_SECONDS + 120))" \
  --closed-loop-measurement-seconds "${MEASURE_SECONDS}" \
  --output-dir "${RUN_DIR}" >"${RUN_DIR}/inference.log" 2>&1

write_state summarizing
RUN_DIR_VALUE="${RUN_DIR}" "${PYTHON}" - <<'PY'
import json, statistics
from pathlib import Path
import os
root = Path(os.environ["RUN_DIR_VALUE"])
rows = [json.loads(line) for line in (root / "affinity-events.jsonl").read_text().splitlines()]
good = [row for row in rows if row["status"] == 200]
local = [row for row in good if row["mode"] == "decode_local_prefill"]
def mean(values):
    return statistics.fmean(values) if values else None
summary = {
    "requests": len(rows), "successful_requests": len(good),
    "local_prefill_requests": len(local),
    "local_prefill_fraction": len(local) / len(good) if good else None,
    "mean_request_seconds": mean([row["elapsed_seconds"] for row in good]),
    "mean_local_request_seconds": mean([row["elapsed_seconds"] for row in local]),
    "mean_local_cached_tokens": mean([row["cached_tokens"] for row in local if row["cached_tokens"] is not None]),
    "mean_local_prompt_tokens": mean([row["prompt_tokens"] for row in local if row["prompt_tokens"] is not None]),
}
(root / "affinity-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

write_state workload_complete
cleanup
trap - EXIT INT TERM

deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
  gpu_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
  port_count=0
  for port in "${ports_to_gate[@]}"; do
    if ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q .; then
      ((port_count+=1))
    fi
  done
  helper_count="$(pgrep -af 'apptainer|squashfuse|fuse-overlayfs' | grep -v "pgrep -af" | wc -l)"
  if (( gpu_count == 0 && port_count == 0 && helper_count == 0 )); then
    write_state complete_clean
    exit 0
  fi
  sleep 5
done
write_state cleanup_gate_failed
exit 1
