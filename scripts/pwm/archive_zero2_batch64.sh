#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_zero2_batch64_20261029
RESULT=docs/results/training_zero2_batch64_20261029
python3 scripts/pwm/verify_zero2_smoke.py "$RUN" "$RESULT"
git add -- "$RESULT"
if ! git diff --cached --quiet; then
  git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: verify all eight ranks and accumulated batch64 updates'
fi
git push origin HEAD:refs/heads/pwm/zero2-batch64
