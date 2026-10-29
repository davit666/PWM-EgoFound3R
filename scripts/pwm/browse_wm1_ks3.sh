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
for prefix in data/ backup_20261023/ ARCTIC/body_models/mano/; do
  name=${prefix//\//_}
  timeout 60s outputs/tools/ks3util -c outputs/secrets/ks3util-config ls "ks3://baai-eai/$prefix" \
    --log-path="$PWD/outputs/ks3-browse/logs" > "$RESULT/ks3_${name}.txt" 2>&1
done
