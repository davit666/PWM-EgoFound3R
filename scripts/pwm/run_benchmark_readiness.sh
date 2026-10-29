#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULT=docs/results/benchmark100_readiness_20261029
mkdir -p "$RESULT"
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/status.json"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: audit current seven-dataset benchmark readiness'
  fi
  git push origin HEAD:refs/heads/pwm/benchmark100-readiness
  exit "$code"
}
trap publish EXIT
python3 scripts/pwm/audit_benchmark_readiness.py "$RESULT/inventory.json" > "$RESULT/run.log" 2>&1
