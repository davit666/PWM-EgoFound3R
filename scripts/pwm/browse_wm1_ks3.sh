#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULT=docs/results/wm1_benchmark100_20261029
mkdir -p "$RESULT" outputs/ks3-browse/logs outputs/ks3-browse/state
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/ks3_browse_status.json"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: inspect KS3 dataset prefixes from wm1'
  fi
  git push origin HEAD:refs/heads/pwm/wm1-benchmark100
  exit "$code"
}
trap publish EXIT
timeout 180s python3 scripts/pwm/browse_wm1_ks3.py "$RESULT/ks3_listing.json" > "$RESULT/ks3_browse.log" 2>&1
