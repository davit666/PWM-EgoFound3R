#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/visual_examples/paper_style_v0}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_ROOT}.status.json}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_visual_demo_v0}"
DEVICE="${DEVICE:-cuda}"

STERA_VIDEO="${STERA_VIDEO:-/share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905/rgb.mp4}"
STERA_START_SECONDS="${STERA_START_SECONDS:-300}"
TACO_ZIP="${TACO_ZIP:-/share/project/eai_pwm/datasets/TACO/Egocentric_RGB_Videos.zip}"
TACO_MEMBER="${TACO_MEMBER:-Egocentric_RGB_Videos/(dust, roller, pan)/20230927_032/color.mp4}"
ARCTIC_ZIP="${ARCTIC_ZIP:-/share/project/eai_pwm/datasets/ARCTIC/downloads/data/images_zips/s01/scissors_use_01.zip}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"

EXAMPLES=(arctic_scissors taco_dust_roller stera_workbench)
FINALIZED=0
TMP_WORK=""

write_status() {
  local state="$1"
  local exit_code="${2:-}"
  local args=("${STATUS_PATH}" "${state}" --video "${STERA_VIDEO}" --output-dir "${OUTPUT_ROOT}")
  if [[ -n "${exit_code}" ]]; then
    args+=(--exit-code "${exit_code}")
  fi
  "${PYTHON_BIN}" scripts/pwm/write_status.py "${args[@]}"
}

cleanup() {
  if [[ -n "${TMP_WORK}" && -d "${TMP_WORK}" ]]; then
    rm -rf -- "${TMP_WORK}"
  fi
}

on_exit() {
  local exit_code=$?
  cleanup
  if [[ "${FINALIZED}" -eq 0 ]]; then
    write_status failed "${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

for command in ffmpeg ffprobe git unzip; do
  command -v "${command}" >/dev/null || {
    printf 'missing required command: %s\n' "${command}" >&2
    exit 2
  }
done
for path in "${PYTHON_BIN}" "${STERA_VIDEO}" "${TACO_ZIP}" "${ARCTIC_ZIP}" \
  checkpoints/EgoFound3R.pt checkpoints/vggt_omega_1b_512.pt; do
  [[ -e "${path}" ]] || {
    printf 'missing required input: %s\n' "${path}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_ROOT}/inputs" "${PUBLISH_DIR}"
TMP_WORK="$(mktemp -d "${OUTPUT_ROOT}/.prepare.XXXXXX")"
write_status running

printf '[visual-demo] preparing 2-second source clips\n'
ffmpeg -hide_banner -loglevel error -y \
  -ss "${STERA_START_SECONDS}" -i "${STERA_VIDEO}" -t 2 -an \
  -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
  "${OUTPUT_ROOT}/inputs/stera_workbench.mp4"

unzip -p "${TACO_ZIP}" "${TACO_MEMBER}" > "${TMP_WORK}/taco_full.mp4"
ffmpeg -hide_banner -loglevel error -y \
  -ss 0.25 -i "${TMP_WORK}/taco_full.mp4" -t 2 -an \
  -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
  "${OUTPUT_ROOT}/inputs/taco_dust_roller.mp4"

mkdir -p "${TMP_WORK}/arctic"
unzip -q "${ARCTIC_ZIP}" '0/*.jpg' -d "${TMP_WORK}/arctic"
ffmpeg -hide_banner -loglevel error -y \
  -framerate 30 -start_number 300 -i "${TMP_WORK}/arctic/0/%05d.jpg" \
  -frames:v 60 -an -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
  "${OUTPUT_ROOT}/inputs/arctic_scissors.mp4"

printf '[visual-demo] validating runtime and checkpoints\n'
"${PYTHON_BIN}" scripts/pwm/preflight.py \
  --video "${OUTPUT_ROOT}/inputs/arctic_scissors.mp4" \
  --config configs/pwm_baseline.toml --require-cuda \
  > "${OUTPUT_ROOT}/preflight.json" 2> "${OUTPUT_ROOT}/preflight.log"

run_example() {
  local name="$1"
  local input_video="${OUTPUT_ROOT}/inputs/${name}.mp4"
  local result_dir="${OUTPUT_ROOT}/${name}"
  printf '[visual-demo] running %s\n' "${name}"
  "${PYTHON_BIN}" infer_marker_video.py \
    --config configs/pwm_baseline.toml \
    --video-path "${input_video}" \
    --checkpoint checkpoints/EgoFound3R.pt \
    --output-dir "${result_dir}" \
    --device "${DEVICE}" \
    --global-stride 5 \
    --chunk-seconds 2.0 \
    --overlap-seconds 0.5 \
    --upsample-mano \
    --overlay-marker-count 778 \
    --no-overlay-frame-images \
    --allow-output-overwrite
  "${PYTHON_BIN}" scripts/pwm/validate_output.py "${result_dir}" \
    > "${result_dir}/validation.json"
}

for example in "${EXAMPLES[@]}"; do
  run_example "${example}"
done

printf '[visual-demo] building compact gallery and montage\n'
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
  "[0:v]scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='ARCTIC - scissors use':x=16:y=10:fontsize=24:fontcolor=white[a];\
   [1:v]scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='TACO - dust / roller / pan':x=16:y=10:fontsize=24:fontcolor=white[b];\
   [2:v]scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='STERA-10M - workbench':x=16:y=10:fontsize=24:fontcolor=white[c];\
   [a][b][c]vstack=inputs=3[gallery]" \
  -map '[gallery]' -frames:v 1 -q:v 2 "${PUBLISH_DIR}/gallery.jpg"

ffmpeg -hide_banner -loglevel error -y \
  -i "${OUTPUT_ROOT}/arctic_scissors/overlay_multimode.mp4" \
  -i "${OUTPUT_ROOT}/taco_dust_roller/overlay_multimode.mp4" \
  -i "${OUTPUT_ROOT}/stera_workbench/overlay_multimode.mp4" \
  -filter_complex \
  "[0:v]fps=15,scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='ARCTIC - scissors use':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[a];\
   [1:v]fps=15,scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='TACO - dust / roller / pan':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[b];\
   [2:v]fps=15,scale=960:960:force_original_aspect_ratio=decrease,pad=960:1000:(ow-iw)/2:40:color=black,drawtext=text='STERA-10M - workbench':x=16:y=10:fontsize=24:fontcolor=white,setsar=1[c];\
   [a][b][c]concat=n=3:v=1:a=0[montage]" \
  -map '[montage]' -an -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
  -movflags +faststart "${PUBLISH_DIR}/montage.mp4"

"${PYTHON_BIN}" scripts/pwm/publish_visual_demo.py \
  --output-root "${OUTPUT_ROOT}" --publish-dir "${PUBLISH_DIR}" \
  "${EXAMPLES[@]}"

cleanup
TMP_WORK=""

if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
  git add -- "${PUBLISH_DIR}"
  if ! git diff --cached --quiet; then
    git commit -m "results: add EgoFound3R paper-style visual examples"
  fi
  git push origin HEAD:main
fi

write_status completed 0
FINALIZED=1
printf '[visual-demo] completed: %s\n' "${PUBLISH_DIR}"
