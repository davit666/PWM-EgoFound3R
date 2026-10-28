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

mkdir -p "${PUBLISH_DIR}"
for example in "${EXAMPLES[@]}"; do
  ffmpeg -hide_banner -loglevel error -y \
    -ss 1 -i "${OUTPUT_ROOT}/${example}/overlay_multimode.mp4" \
    -frames:v 1 -q:v 2 "${PUBLISH_DIR}/${example}.jpg"
done

ffmpeg -hide_banner -loglevel error -y \
  -i "${PUBLISH_DIR}/arctic_scissors.jpg" \
  -i "${PUBLISH_DIR}/taco_dust_roller.jpg" \
  -i "${PUBLISH_DIR}/stera_workbench.jpg" \
  -filter_complex \
  "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='ARCTIC - scissors use':x=16:y=10:fontsize=24:fontcolor=white[a];\
   [1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='TACO - dust / roller / pan':x=16:y=10:fontsize=24:fontcolor=white[b];\
   [2:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='STERA-10M - workbench':x=16:y=10:fontsize=24:fontcolor=white[c];\
   [a][b][c]vstack=inputs=3[gallery]" \
  -map '[gallery]' -frames:v 1 -q:v 2 "${PUBLISH_DIR}/gallery.jpg"

ffmpeg -hide_banner -loglevel error -y \
  -i "${OUTPUT_ROOT}/arctic_scissors/overlay_multimode.mp4" \
  -i "${OUTPUT_ROOT}/taco_dust_roller/overlay_multimode.mp4" \
  -i "${OUTPUT_ROOT}/stera_workbench/overlay_multimode.mp4" \
  -filter_complex \
  "[0:v]fps=15,scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='ARCTIC - scissors use':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[a];\
   [1:v]fps=15,scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='TACO - dust / roller / pan':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[b];\
   [2:v]fps=15,scale=960:540:force_original_aspect_ratio=decrease,pad=960:580:(ow-iw)/2:40:color=black,drawtext=text='STERA-10M - workbench':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[c];\
   [a][b][c]concat=n=3:v=1:a=0[montage]" \
  -map '[montage]' -an -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
  -movflags +faststart "${PUBLISH_DIR}/montage.mp4"

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
