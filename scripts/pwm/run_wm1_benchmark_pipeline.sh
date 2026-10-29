#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for phase in stage prepare-stera prepare-taco train evaluate; do
  bash scripts/pwm/run_wm1_benchmark100.sh "$phase"
done
