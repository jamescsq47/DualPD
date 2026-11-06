#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?usage: import_single_uid_image.sh IMAGE [TAG]}"
TAG="${2:-${IMAGE}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRANE_BIN="${CRANE_BIN:-/orcd/scratch/orcd/003/shangy/dualpd/tools/crane-v0.22.1/crane}"
DOCKER_BIN="${DOCKER_BIN:-/orcd/scratch/orcd/003/shangy/dualpd/tools/docker-rootless-28.5.2/bin/docker}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

work_dir="$(mktemp -d /tmp/dualpd-singleuid-image.XXXXXX)"
cleanup() {
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT

"${CRANE_BIN}" config "${IMAGE}" >"${work_dir}/config.json"
"${CRANE_BIN}" export "${IMAGE}" "${work_dir}/rootfs.tar"
"${PYTHON_BIN}" "${SCRIPT_DIR}/normalize_tar_owners.py" \
  "${work_dir}/rootfs.tar" "${work_dir}/rootfs-uid0.tar"

changes=()
while IFS= read -r value; do
  [[ -n "${value}" ]] && changes+=(--change "ENV ${value}")
done < <(jq -r '.config.Env[]?' "${work_dir}/config.json")

workdir="$(jq -r '.config.WorkingDir // empty' "${work_dir}/config.json")"
user="$(jq -r '.config.User // empty' "${work_dir}/config.json")"
entrypoint="$(jq -c '.config.Entrypoint // empty' "${work_dir}/config.json")"
cmd="$(jq -c '.config.Cmd // empty' "${work_dir}/config.json")"
[[ -n "${workdir}" ]] && changes+=(--change "WORKDIR ${workdir}")
[[ -n "${user}" ]] && changes+=(--change "USER ${user}")
[[ -n "${entrypoint}" ]] && changes+=(--change "ENTRYPOINT ${entrypoint}")
[[ -n "${cmd}" ]] && changes+=(--change "CMD ${cmd}")

"${DOCKER_BIN}" import "${changes[@]}" "${work_dir}/rootfs-uid0.tar" "${TAG}"
"${DOCKER_BIN}" image inspect "${TAG}" \
  --format '{{.Id}} {{json .Config.Env}} {{json .Config.WorkingDir}} {{json .Config.User}}'
