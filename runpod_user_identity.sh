#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POD_ID_FILE="${SCRIPT_DIR}/.runpod_pod_id"

# Config defaults (override via env vars)
POD_NAME="${RUNPOD_POD_NAME:-user-identity-mechinterp}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-${RUNPOD_GPU_TYPE_ID:-NVIDIA GeForce RTX 4090}}"
IMAGE_NAME="${RUNPOD_IMAGE_NAME:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-ALL}"
CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
VOLUME_GB="${RUNPOD_VOLUME_GB:-50}"
MIN_VCPU="${RUNPOD_MIN_VCPU:-8}"
MIN_MEM_GB="${RUNPOD_MIN_MEM_GB:-30}"
PORTS="${RUNPOD_PORTS:-22/tcp}"

usage() {
  cat <<'USAGE'
Usage: runpod_user_identity.sh <create|ssh|status|terminate|help>

Environment variables:
  RUNPOD_API_KEY        (required)
  RUNPOD_GPU_TYPE       (default: NVIDIA GeForce RTX 4090)
  RUNPOD_GPU_TYPE_ID    (accepted for backward compatibility)
  RUNPOD_POD_NAME       (default: user-identity-mechinterp)
  RUNPOD_IMAGE_NAME     (default: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04)
  RUNPOD_CLOUD_TYPE     (default: ALL; options: ALL, SECURE, COMMUNITY)
  RUNPOD_CONTAINER_DISK_GB (default: 50)
  RUNPOD_VOLUME_GB      (default: 50)
  RUNPOD_MIN_VCPU       (default: 8)
  RUNPOD_MIN_MEM_GB     (default: 30)
  RUNPOD_PORTS          (default: 22/tcp)
  RUNPOD_ENV            (optional, comma-separated KEY=VALUE pairs)

Notes:
  - This script uses runpodctl. Install it first and add it to PATH.
  - Add your SSH public key in Runpod account settings before creating a pod.
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found in PATH. Install it first." >&2
    exit 1
  fi
}

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "Error: $1 is required." >&2
    exit 1
  fi
}

require_gpu_type() {
  if [[ -z "${GPU_TYPE:-}" ]]; then
    echo "Error: RUNPOD_GPU_TYPE is required." >&2
    exit 1
  fi
}

create_pod() {
  require_cmd runpodctl
  require_env RUNPOD_API_KEY
  require_gpu_type

  export RUNPOD_API_KEY

  echo "Creating Runpod pod..."
  args=(
    --name "${POD_NAME}"
    --gpuType "${GPU_TYPE}"
    --imageName "${IMAGE_NAME}"
    --containerDiskSize "${CONTAINER_DISK_GB}"
    --volumeSize "${VOLUME_GB}"
    --vcpu "${MIN_VCPU}"
    --mem "${MIN_MEM_GB}"
    --ports "${PORTS}"
  )
  if [[ -n "${RUNPOD_ENV:-}" ]]; then
    IFS=',' read -r -a env_pairs <<< "${RUNPOD_ENV}"
    for pair in "${env_pairs[@]}"; do
      args+=(--env "${pair}")
    done
  fi
  if [[ "${CLOUD_TYPE}" == "SECURE" ]]; then
    args+=(--secureCloud)
  elif [[ "${CLOUD_TYPE}" == "COMMUNITY" ]]; then
    args+=(--communityCloud)
  fi
  runpodctl create pod "${args[@]}"

  echo ""
  echo "Paste the pod ID from the output above."
  read -r pod_id
  if [[ -z "${pod_id}" ]]; then
    echo "Error: pod ID is required." >&2
    exit 1
  fi
  echo "${pod_id}" > "${POD_ID_FILE}"
  echo "Saved pod ID to ${POD_ID_FILE}"
}

ssh_pod() {
  require_cmd runpodctl
  if [[ ! -f "${POD_ID_FILE}" ]]; then
    echo "Error: ${POD_ID_FILE} not found. Run create first." >&2
    exit 1
  fi
  runpodctl ssh "$(cat "${POD_ID_FILE}")"
}

status_pod() {
  require_cmd runpodctl
  if [[ ! -f "${POD_ID_FILE}" ]]; then
    echo "Error: ${POD_ID_FILE} not found. Run create first." >&2
    exit 1
  fi
  runpodctl get pod "$(cat "${POD_ID_FILE}")"
}

terminate_pod() {
  require_cmd runpodctl
  if [[ ! -f "${POD_ID_FILE}" ]]; then
    echo "Error: ${POD_ID_FILE} not found. Run create first." >&2
    exit 1
  fi
  runpodctl terminate pod "$(cat "${POD_ID_FILE}")"
  rm -f "${POD_ID_FILE}"
}

cmd="${1:-help}"
case "${cmd}" in
  create)
    create_pod
    ;;
  ssh)
    ssh_pod
    ;;
  status)
    status_pod
    ;;
  terminate)
    terminate_pod
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
 esac
