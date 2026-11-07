#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A compute-node Docker daemon for clusters that enable unprivileged user
# namespaces but do not install Docker or allocate /etc/subuid ranges.  The
# daemon itself is real dockerd; it runs as uid 0 only inside the caller's
# single-uid user namespace.  VFS is intentional because overlay-on-overlay
# is not portable in this mode.

DOCKER_STATIC_ROOT="${DOCKER_STATIC_ROOT:-/orcd/scratch/orcd/003/shangy/dualpd/tools/docker-rootless-28.5.2/bin}"
DOCKER_RUNTIME_DIR="${DOCKER_RUNTIME_DIR:-/tmp/dualpd-docker-${SLURM_JOB_ID:-manual}}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/orcd/scratch/orcd/003/shangy/dualpd/docker/job-${SLURM_JOB_ID:-manual}}"
ROOTLESSKIT_SINGLE_UID_BIN="${ROOTLESSKIT_SINGLE_UID_BIN:-${DOCKER_STATIC_ROOT}/rootlesskit-singleuid}"
FUSE_OVERLAYFS_ROOT="${FUSE_OVERLAYFS_ROOT:-/orcd/scratch/orcd/003/shangy/dualpd/tools/fuse-overlayfs-v1.17}"
DOCKER_STORAGE_DRIVER="${DOCKER_STORAGE_DRIVER:-fuse-overlayfs}"

if [[ "${1:-}" != --child ]]; then
  state_dir="${DOCKER_RUNTIME_DIR}/rootlesskit"
  mkdir -p "${DOCKER_RUNTIME_DIR}" "${DOCKER_DATA_ROOT}" "${state_dir}"
  chmod 700 "${DOCKER_RUNTIME_DIR}"
  [[ -x "${ROOTLESSKIT_SINGLE_UID_BIN}" ]] || {
    echo "single-UID RootlessKit binary is missing: ${ROOTLESSKIT_SINGLE_UID_BIN}" >&2
    exit 1
  }
  export PATH="${SCRIPT_DIR}/rootless-bin:${DOCKER_STATIC_ROOT}:${FUSE_OVERLAYFS_ROOT}:${PATH}"
  export ROOTLESSKIT_SINGLE_UID=1
  exec "${ROOTLESSKIT_SINGLE_UID_BIN}" \
    --state-dir="${state_dir}" \
    --net=slirp4netns --mtu=65520 --port-driver=builtin \
    --copy-up=/etc --copy-up=/run --propagation=rslave \
    "$0" --child
fi

export PATH="${SCRIPT_DIR}/rootless-bin:${DOCKER_STATIC_ROOT}:${FUSE_OVERLAYFS_ROOT}:${PATH}"
export XDG_RUNTIME_DIR="${DOCKER_RUNTIME_DIR}"
exec dockerd \
  --host="unix://${DOCKER_RUNTIME_DIR}/docker.sock" \
  --data-root="${DOCKER_DATA_ROOT}" \
  --exec-root="${DOCKER_RUNTIME_DIR}/exec" \
  --pidfile="${DOCKER_RUNTIME_DIR}/dockerd.pid" \
  --bridge=none \
  --iptables=false \
  --ip6tables=false \
  --ip-forward=false \
  --ip-masq=false \
  --exec-opt native.cgroupdriver=cgroupfs \
  --storage-driver="${DOCKER_STORAGE_DRIVER}"
