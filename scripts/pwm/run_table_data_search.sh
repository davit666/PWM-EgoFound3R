#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULT=docs/results/table_data_search_20261029
mkdir -p "$RESULT"
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/status.json"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: search dataset locations from the inventory spreadsheet'
  fi
  git push origin HEAD:refs/heads/pwm/table-data-search
  exit "$code"
}
trap publish EXIT
python3 scripts/pwm/locate_table_sources.py "$RESULT/search.json" > "$RESULT/run.log" 2>&1
