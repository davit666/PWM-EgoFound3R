#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VIDEO_PATH="${1:-${VIDEO_PATH:-}}"
if [[ -z "${VIDEO_PATH}" ]]; then
  printf 'usage: %s /absolute/path/to/video.mp4\n' "$0" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pwm_baseline/smoke}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_DIR}.status.json}"
DEVICE="${DEVICE:-cuda}"
MAX_FRAMES="${MAX_FRAMES:-20}"
FINALIZED=0

on_exit() {
  local exit_code=$?
  if [[ "${FINALIZED}" -eq 0 ]]; then
    "${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" failed \
      --video "${VIDEO_PATH}" --output-dir "${OUTPUT_DIR}" --exit-code "${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

"${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" running \
  --video "${VIDEO_PATH}" --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/pwm/preflight.py \
  --video "${VIDEO_PATH}" --config configs/pwm_baseline.toml --require-cuda

overwrite_args=()
if [[ "${ALLOW_OUTPUT_OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--allow-output-overwrite)
fi

"${PYTHON_BIN}" infer_marker_video.py \
  --config configs/pwm_baseline.toml \
  --video-path "${VIDEO_PATH}" \
  --checkpoint checkpoints/EgoFound3R.pt \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --max-frames "${MAX_FRAMES}" \
  --chunk-seconds 2.0 \
  --overlap-seconds 0.5 \
  --upsample-mano \
  --no-overlay-frame-images \
  "${overwrite_args[@]}"

"${PYTHON_BIN}" scripts/pwm/validate_output.py "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/validation.json"
"${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" completed \
  --video "${VIDEO_PATH}" --output-dir "${OUTPUT_DIR}" --exit-code 0
FINALIZED=1
