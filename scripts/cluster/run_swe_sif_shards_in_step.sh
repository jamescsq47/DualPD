#!/usr/bin/env bash
set -euo pipefail

# Run one Slurm task per node. Each task launches a small, disjoint set of
# CPU-only conversion workers; GPUs remain untouched.
: "${SLURM_JOB_ID:?must run inside a Slurm allocation}"
: "${SLURM_PROCID:?must run as an srun task}"
: "${SLURM_NTASKS:?must run as an srun task}"

workers_per_node="${SWE_SIF_WORKERS_PER_NODE:-4}"
log_tag="${SWE_SIF_LOG_TAG:-${SLURM_JOB_ID}}"
temporary_base="${SWE_SIF_TEMP_BASE:-/tmp/shangy}"
sif_dir="${SWE_SIF_DIR:-/orcd/scratch/orcd/003/shangy/dualpd/containers/swebench-verified-sif}"
if [[ ! "${workers_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SWE_SIF_WORKERS_PER_NODE must be positive" >&2
  exit 2
fi
if [[ "${temporary_base}" != /* ]]; then
  echo "SWE_SIF_TEMP_BASE must be an absolute path" >&2
  exit 2
fi

mkdir -p "${temporary_base}"
mkdir -p "${sif_dir}"

total_shards=$((SLURM_NTASKS * workers_per_node))
pids=()
for ((local_worker = 0; local_worker < workers_per_node; local_worker++)); do
  shard=$((SLURM_PROCID + SLURM_NTASKS * local_worker))
  temporary_root="${temporary_base}/pd-sif-convert-${log_tag}-${shard}"
  stdout="/home/shangy/dualpd/swe-sif-step-${log_tag}-${shard}.out"
  stderr="/home/shangy/dualpd/swe-sif-step-${log_tag}-${shard}.err"
  python /home/shangy/dualpd/DualPD/scripts/cluster/convert_swe_oci_to_sif.py \
    --dataset /orcd/scratch/orcd/003/shangy/dualpd/datasets/swe-bench-verified/test.jsonl \
    --oci-layout /orcd/scratch/orcd/003/shangy/dualpd/containers/swebench-verified-oci \
    --sif-dir "${sif_dir}" \
    --temporary-root "${temporary_root}" \
    --shard-count "${total_shards}" \
    --shard-index "${shard}" \
    --skopeo /usr/bin/skopeo \
    --policy /home/shangy/dualpd/DualPD/scripts/cluster/skopeo-public-policy.json \
    --mksquashfs-processors 4 \
    >"${stdout}" 2>"${stderr}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
