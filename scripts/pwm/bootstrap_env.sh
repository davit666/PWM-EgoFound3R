#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.txt"
if ! "${VENV_DIR}/bin/python" -m pip check; then
  printf '%s\n' \
    'WARNING: inherited system packages have dependency conflicts; project imports are validated by preflight.py.' \
    >&2
fi
printf 'PWM_PYTHON=%s\n' "${VENV_DIR}/bin/python"
