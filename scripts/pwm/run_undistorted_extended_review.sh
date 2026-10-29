#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_ROOT="${BASE_ROOT:-${REPO_ROOT}/outputs/8task_review_v3_undistorted_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/8task_review_v3_undistorted}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_8task_review_v3_undistorted}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"

REQUIRE_UNDISTORTION=1 \
OUTPUT_ROOT="${BASE_ROOT}" \
PUBLISH_DIR="${BASE_ROOT}/published_2x2" \
PUBLISH_RESULTS=0 \
bash scripts/pwm/run_8task_visual_review.sh

BASE_ROOT="${BASE_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
PUBLISH_DIR="${PUBLISH_DIR}" \
PUBLISH_RESULTS="${PUBLISH_RESULTS}" \
bash scripts/pwm/run_extended_mano_review.sh
