#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
GPU_INDEX="${GPU_INDEX:-0}"
CLIP_SECONDS="${CLIP_SECONDS:-8}"
TARGET_FPS="${TARGET_FPS:-10}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
TASK_COUNT="${TASK_COUNT:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/8task_review_v0}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_ROOT}/status.json}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_8task_review_v0}"
GUOCAI_ROOT="${GUOCAI_ROOT:-/share-mixed/projset/eai_pwm/datasets/ours-ego-sampled/guocai-ego}"
JIANZHI_ROOT="${JIANZHI_ROOT:-/share-mixed/projset/eai_pwm/datasets/ours-ego-sampled/jianzhi-ego}"
STERA_ROOT="${STERA_ROOT:-/share-mixed/projset/eai_pwm/datasets/Stera-10M}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/EgoFound3R.pt}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/pwm_baseline.toml}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
REQUIRE_UNDISTORTION="${REQUIRE_UNDISTORTION:-0}"
GIT_RESULT_AUTHOR_NAME="${GIT_RESULT_AUTHOR_NAME:-Wenhao Li}"
GIT_RESULT_AUTHOR_EMAIL="${GIT_RESULT_AUTHOR_EMAIL:-wl2655@columbia.edu}"
FINALIZED=0

write_status() {
  "${PYTHON}" scripts/pwm/write_job_status.py "${STATUS_PATH}" "$@"
}

on_exit() {
  local exit_code=$?
  if [[ "${FINALIZED}" -eq 0 ]]; then
    write_status failed "exit_code=${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT INT TERM

for command in ffmpeg ffprobe git nvidia-smi; do
  command -v "${command}" >/dev/null || {
    printf 'missing required command: %s\n' "${command}" >&2
    exit 2
  }
done
for path in "${PYTHON}" "${GUOCAI_ROOT}" "${JIANZHI_ROOT}" "${STERA_ROOT}" \
  "${CHECKPOINT}" "${CONFIG}" checkpoints/vggt_omega_1b_512.pt; do
  [[ -e "${path}" ]] || {
    printf 'missing required input: %s\n' "${path}" >&2
    exit 2
  }
done
case "${OUTPUT_ROOT}" in
  /share-mixed/projset/*|/share-mixed/project/*|/home-mixed/liwenhao666/*) ;;
  *) printf 'refusing non-persistent output root: %s\n' "${OUTPUT_ROOT}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
mkdir -p "${OUTPUT_ROOT}/inputs" "${OUTPUT_ROOT}/inference" "${PUBLISH_DIR}"
write_status preparing \
  "task_count=${TASK_COUNT}" "clip_seconds=${CLIP_SECONDS}" \
  "target_fps=${TARGET_FPS}" "gpu_index=${GPU_INDEX}" \
  "require_undistortion=${REQUIRE_UNDISTORTION}"

undistortion_arguments=()
if [[ "${REQUIRE_UNDISTORTION}" == "1" ]]; then
  undistortion_arguments+=(--require-undistortion)
fi

printf '[8task] preparing Guocai duration-matched tasks\n'
"${PYTHON}" scripts/pwm/prepare_task_montage.py \
  --dataset guocai --root "${GUOCAI_ROOT}" \
  --output "${OUTPUT_ROOT}/inputs/guocai" \
  --task-count "${TASK_COUNT}" --max-clip-seconds "${CLIP_SECONDS}" \
  --min-clip-seconds 2 --target-fps "${TARGET_FPS}" \
  --width "${WIDTH}" --height "${HEIGHT}" --overwrite \
  "${undistortion_arguments[@]}" \
  > "${OUTPUT_ROOT}/prepare-guocai.log" 2>&1

printf '[8task] preparing Jianzhi duration-matched tasks from MCAP camera2\n'
"${PYTHON}" scripts/pwm/prepare_jianzhi_montage.py \
  --root "${JIANZHI_ROOT}" --output "${OUTPUT_ROOT}/inputs/jianzhi" \
  --topic /robot0/sensor/camera2/compressed \
  --task-count "${TASK_COUNT}" --clip-seconds "${CLIP_SECONDS}" \
  --target-fps "${TARGET_FPS}" --width "${WIDTH}" --height "${HEIGHT}" \
  --overwrite "${undistortion_arguments[@]}" \
  > "${OUTPUT_ROOT}/prepare-jianzhi.log" 2>&1

printf '[8task] preparing Stera-10M duration-matched tasks\n'
"${PYTHON}" scripts/pwm/prepare_task_montage.py \
  --dataset stera --root "${STERA_ROOT}" \
  --output "${OUTPUT_ROOT}/inputs/stera" \
  --task-count "${TASK_COUNT}" --max-clip-seconds "${CLIP_SECONDS}" \
  --min-clip-seconds 2 --target-fps "${TARGET_FPS}" \
  --width "${WIDTH}" --height "${HEIGHT}" --overwrite \
  "${undistortion_arguments[@]}" \
  > "${OUTPUT_ROOT}/prepare-stera.log" 2>&1

"${PYTHON}" scripts/pwm/preflight.py \
  --config "${CONFIG}" --require-cuda --skip-weight-hash \
  --video "${OUTPUT_ROOT}/inputs/guocai/guocai_8tasks_source.mp4" \
  > "${OUTPUT_ROOT}/preflight.json"

run_dataset() {
  local dataset="$1"
  local source_video="${OUTPUT_ROOT}/inputs/${dataset}/${dataset}_8tasks_source.mp4"
  local inference_dir="${OUTPUT_ROOT}/inference/${dataset}"
  local published_video="${PUBLISH_DIR}/${dataset}_8tasks.mp4"
  write_status running "stage=inference" "current_dataset=${dataset}"
  printf '[8task] inference: %s\n' "${dataset}"
  "${PYTHON}" infer_marker_video.py \
    --config "${CONFIG}" \
    --video-path "${source_video}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${inference_dir}" \
    --device "${DEVICE}" \
    --global-stride 5 \
    --chunk-seconds 2.0 \
    --overlap-seconds 0.5 \
    --overlay-marker-count 195 \
    --no-overlay-frame-images \
    --allow-output-overwrite \
    > "${OUTPUT_ROOT}/infer-${dataset}.log" 2>&1
  "${PYTHON}" scripts/pwm/validate_output.py "${inference_dir}" \
    > "${inference_dir}/validation.json"
  ffmpeg -hide_banner -loglevel error -y \
    -i "${inference_dir}/overlay_multimode.mp4" -an \
    -vf "fps=${TARGET_FPS},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
    -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart \
    "${published_video}"
  cp "${OUTPUT_ROOT}/inputs/${dataset}/manifest.json" \
    "${PUBLISH_DIR}/${dataset}_manifest.json"
  ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
    -show_entries format=duration,size -of json "${published_video}" \
    > "${PUBLISH_DIR}/${dataset}_media.json"
  printf '[8task] completed: %s\n' "${dataset}"
}

for dataset in guocai jianzhi stera; do
  run_dataset "${dataset}"
done

printf "file '%s'\nfile '%s'\nfile '%s'\n" \
  "${PUBLISH_DIR}/guocai_8tasks.mp4" \
  "${PUBLISH_DIR}/jianzhi_8tasks.mp4" \
  "${PUBLISH_DIR}/stera_8tasks.mp4" \
  > "${OUTPUT_ROOT}/published-concat.txt"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 \
  -i "${OUTPUT_ROOT}/published-concat.txt" -an -c copy -movflags +faststart \
  "${PUBLISH_DIR}/all_three_24tasks.mp4"
combined_bytes="$(stat -c '%s' "${PUBLISH_DIR}/all_three_24tasks.mp4")"
if (( combined_bytes >= 98000000 )); then
  rm -f "${PUBLISH_DIR}/all_three_24tasks.mp4"
fi

while IFS= read -r -d '' video; do
  video_bytes="$(stat -c '%s' "${video}")"
  if (( video_bytes >= 98000000 )); then
    printf 'published video exceeds safe GitHub size: %s (%s bytes)\n' "${video}" "${video_bytes}" >&2
    exit 3
  fi
done < <(find "${PUBLISH_DIR}" -maxdepth 1 -type f -name '*.mp4' -print0)

if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
  git add "${PUBLISH_DIR}"
  if ! git diff --cached --quiet; then
    git -c user.name="${GIT_RESULT_AUTHOR_NAME}" \
      -c user.email="${GIT_RESULT_AUTHOR_EMAIL}" \
      commit -m "results: add three eight-task EgoFound3R reviews"
  fi
  git push origin HEAD:main
fi

write_status completed "exit_code=0" "stage=published" "current_dataset=all"
FINALIZED=1
printf '[8task] completed and published: %s\n' "${PUBLISH_DIR}"
