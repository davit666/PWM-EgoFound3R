#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $#==0 )); then set -- stage prepare-stera prepare-taco train evaluate; fi
for phase in "$@"; do
  bash scripts/pwm/run_wm1_benchmark100.sh "$phase"
done
