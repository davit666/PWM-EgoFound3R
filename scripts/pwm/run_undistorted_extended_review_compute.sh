#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

case "${REPO_ROOT}" in
  /home-mixed/liwenhao666/*|/share-mixed/project/*) ;;
  *) printf 'refusing non-persistent compute checkout: %s\n' "${REPO_ROOT}" >&2; exit 2 ;;
esac

RUN_ROOT="${RUN_ROOT:-/share-mixed/project/eai_pwm/PWM-EgoFound3R}"
case "${RUN_ROOT}" in
  /share-mixed/project/*) ;;
  *) printf 'refusing non-persistent run root: %s\n' "${RUN_ROOT}" >&2; exit 2 ;;
esac

export BASE_ROOT="${BASE_ROOT:-${RUN_ROOT}/outputs/8task_review_v3_undistorted_base}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/outputs/8task_review_v3_undistorted}"
export PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_8task_review_v3_undistorted}"
export PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
export GPU_INDEX="${GPU_INDEX:-0}"

exec bash scripts/pwm/run_undistorted_extended_review.sh
