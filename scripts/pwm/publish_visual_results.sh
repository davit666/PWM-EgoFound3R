#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/visual_examples/paper_style_v0}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_ROOT}.status.json}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_visual_demo_v0}"
STERA_VIDEO="${STERA_VIDEO:-/share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905/rgb.mp4}"
GIT_RESULT_AUTHOR_NAME="${GIT_RESULT_AUTHOR_NAME:-Wenhao Li}"
GIT_RESULT_AUTHOR_EMAIL="${GIT_RESULT_AUTHOR_EMAIL:-wl2655@columbia.edu}"
EXAMPLES=(arctic_scissors taco_dust_roller stera_workbench)

for example in "${EXAMPLES[@]}"; do
  for artifact in summary.json validation.json overlay.mp4 overlay_multimode.mp4; do
    path="${OUTPUT_ROOT}/${example}/${artifact}"
    [[ -s "${path}" ]] || {
      printf 'missing result artifact: %s\n' "${path}" >&2
      exit 2
    }
  done
  "${PYTHON_BIN}" scripts/pwm/validate_output.py "${OUTPUT_ROOT}/${example}" \
    > "${OUTPUT_ROOT}/${example}/validation.json"
done

for artifact in README.md manifest.json gallery.jpg montage.mp4; do
  path="${PUBLISH_DIR}/${artifact}"
  [[ -s "${path}" ]] || {
    printf 'missing publish artifact: %s\n' "${path}" >&2
    exit 2
  }
done

"${PYTHON_BIN}" scripts/pwm/publish_visual_demo.py \
  --output-root "${OUTPUT_ROOT}" --publish-dir "${PUBLISH_DIR}" \
  "${EXAMPLES[@]}"

git add -- "${PUBLISH_DIR}"
if ! git diff --cached --quiet; then
  git -c user.name="${GIT_RESULT_AUTHOR_NAME}" \
    -c user.email="${GIT_RESULT_AUTHOR_EMAIL}" \
    commit -m "results: add EgoFound3R paper-style visual examples"
fi
git push origin HEAD:main

"${PYTHON_BIN}" scripts/pwm/write_status.py "${STATUS_PATH}" completed \
  --video "${STERA_VIDEO}" --output-dir "${OUTPUT_ROOT}" --exit-code 0
printf '[visual-demo] published existing results: %s\n' "${PUBLISH_DIR}"
