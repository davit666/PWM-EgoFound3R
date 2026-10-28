#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/outputs/checkpoint_download.log}"
STATUS_PATH="${STATUS_PATH:-${REPO_ROOT}/outputs/checkpoint_download.status.json}"
mkdir -p "$(dirname "${LOG_PATH}")"

"${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" running \
  --output-dir "${REPO_ROOT}/checkpoints"

set +e
"${PYTHON_BIN}" scripts/pwm/download_checkpoints.py >"${LOG_PATH}" 2>&1
exit_code=$?
set -e

if [[ "${exit_code}" -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" completed \
    --output-dir "${REPO_ROOT}/checkpoints" --exit-code 0
else
  "${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" failed \
    --output-dir "${REPO_ROOT}/checkpoints" --exit-code "${exit_code}"
fi
exit "${exit_code}"
