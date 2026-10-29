#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULT_DIR=docs/results/training_source_audit_20261029
mkdir -p "$RESULT_DIR"
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT_DIR/status.json"
  git add -- "$RESULT_DIR"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: audit training source layouts'
  fi
  git push origin HEAD:refs/heads/pwm/training-source-audit
  exit "$code"
}
trap publish EXIT
PYTHON_BIN=/home-mixed/liwenhao666/wm/PWM-EgoFound3R/.venv/bin/python
"$PYTHON_BIN" scripts/pwm/audit_training_sources.py "$RESULT_DIR/inventory.json" > "$RESULT_DIR/run.log" 2>&1
if [[ "${ANNOTATIONS:-0}" == 1 ]]; then
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps --no-deps h5py > "$RESULT_DIR/dependencies.log" 2>&1
  export PYTHONPATH="$PWD/outputs/training-deps${PYTHONPATH:+:$PYTHONPATH}"
  "$PYTHON_BIN" scripts/pwm/audit_training_annotations.py "$RESULT_DIR/annotations.json" >> "$RESULT_DIR/run.log" 2>&1
fi
