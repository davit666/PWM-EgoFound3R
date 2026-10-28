#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
BASE_ROOT="${BASE_ROOT:-${REPO_ROOT}/outputs/8task_review_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/8task_review_v1_extended}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/docs/results/egofound3r_8task_review_v1_extended}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_ROOT}/status.json}"
DATASETS="${DATASETS:-guocai jianzhi stera}"
MAX_FRAMES="${MAX_FRAMES:-}"
CRF="${CRF:-25}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
GIT_RESULT_AUTHOR_NAME="${GIT_RESULT_AUTHOR_NAME:-Wenhao Li}"
GIT_RESULT_AUTHOR_EMAIL="${GIT_RESULT_AUTHOR_EMAIL:-wl2655@columbia.edu}"
FINALIZED=0

write_status() {
  "${PYTHON}" scripts/pwm/write_job_status.py "${STATUS_PATH}" "$@"
}

publish_failure_diagnostics() {
  local exit_code="$1"
  local diagnostics="${PUBLISH_DIR}/diagnostics"
  mkdir -p "${diagnostics}"
  cp "${STATUS_PATH}" "${diagnostics}/status.json" 2>/dev/null || true
  for log in "${OUTPUT_ROOT}"/*.log; do
    [[ -f "${log}" ]] || continue
    tail -200 "${log}" > "${diagnostics}/$(basename "${log}").tail.txt" || true
  done
  printf 'exit_code=%s\n' "${exit_code}" > "${diagnostics}/exit-code.txt"
  if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
    git add "${diagnostics}" || true
    if ! git diff --cached --quiet; then
      git -c user.name="${GIT_RESULT_AUTHOR_NAME}" \
        -c user.email="${GIT_RESULT_AUTHOR_EMAIL}" \
        commit -m "diagnostics: archive extended MANO review failure" || true
    fi
    git push origin HEAD:main || true
  fi
}

on_exit() {
  local exit_code=$?
  if [[ "${FINALIZED}" -eq 0 ]]; then
    write_status failed "exit_code=${exit_code}" || true
    publish_failure_diagnostics "${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT INT TERM

for command in ffmpeg ffprobe git; do
  command -v "${command}" >/dev/null || {
    printf 'missing required command: %s\n' "${command}" >&2
    exit 2
  }
done
for path in "${PYTHON}" "${BASE_ROOT}"; do
  [[ -e "${path}" ]] || {
    printf 'missing required input: %s\n' "${path}" >&2
    exit 2
  }
done
case "${OUTPUT_ROOT}" in
  /share/mixed/projset/*|/share-mixed/projset/*) ;;
  *) printf 'refusing non-mixed output root: %s\n' "${OUTPUT_ROOT}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_ROOT}" "${PUBLISH_DIR}"
write_status preparing "datasets=${DATASETS}" "layout=2x2+1x2+rgb" "crf=${CRF}"

for dataset in ${DATASETS}; do
  multimode="${BASE_ROOT}/inference/${dataset}/overlay_multimode.mp4"
  rgb="${BASE_ROOT}/inputs/${dataset}/${dataset}_8tasks_source.mp4"
  predictions="${BASE_ROOT}/inference/${dataset}/inference_output.pt"
  output="${PUBLISH_DIR}/${dataset}_8tasks_extended.mp4"
  layout_manifest="${PUBLISH_DIR}/${dataset}_layout_manifest.json"
  for path in "${multimode}" "${rgb}" "${predictions}"; do
    [[ -s "${path}" ]] || {
      printf 'missing or empty base artifact: %s\n' "${path}" >&2
      exit 2
    }
  done
  write_status running "current_dataset=${dataset}" "stage=compose"
  arguments=(
    --multimode-video "${multimode}"
    --rgb-video "${rgb}"
    --inference-output "${predictions}"
    --output "${output}"
    --manifest "${layout_manifest}"
    --fps 10
    --crf "${CRF}"
  )
  if [[ -n "${MAX_FRAMES}" ]]; then
    arguments+=(--max-frames "${MAX_FRAMES}")
  fi
  "${PYTHON}" scripts/pwm/compose_extended_mano_layout.py "${arguments[@]}" \
    > "${OUTPUT_ROOT}/compose-${dataset}.log" 2>&1
  ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
    -show_entries format=duration,size -of json "${output}" \
    > "${PUBLISH_DIR}/${dataset}_media.json"
  ffmpeg -hide_banner -loglevel error -i "${output}" -f null -
  if [[ -f "docs/results/egofound3r_8task_review_v0/${dataset}_manifest.json" ]]; then
    cp "docs/results/egofound3r_8task_review_v0/${dataset}_manifest.json" \
      "${PUBLISH_DIR}/${dataset}_task_manifest.json"
  fi
  printf '[extended-mano] completed: %s\n' "${dataset}"
done

if [[ -f "${PUBLISH_DIR}/guocai_8tasks_extended.mp4" \
   && -f "${PUBLISH_DIR}/jianzhi_8tasks_extended.mp4" \
   && -f "${PUBLISH_DIR}/stera_8tasks_extended.mp4" ]]; then
  printf "file '%s'\nfile '%s'\nfile '%s'\n" \
    "${PUBLISH_DIR}/guocai_8tasks_extended.mp4" \
    "${PUBLISH_DIR}/jianzhi_8tasks_extended.mp4" \
    "${PUBLISH_DIR}/stera_8tasks_extended.mp4" \
    > "${OUTPUT_ROOT}/published-concat.txt"
  ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 \
    -i "${OUTPUT_ROOT}/published-concat.txt" -an -c copy -movflags +faststart \
    "${PUBLISH_DIR}/all_three_24tasks_extended.mp4"
  combined_bytes="$(stat -c '%s' "${PUBLISH_DIR}/all_three_24tasks_extended.mp4")"
  if (( combined_bytes >= 98000000 )); then
    rm -f "${PUBLISH_DIR}/all_three_24tasks_extended.mp4"
  fi
fi

while IFS= read -r -d '' video; do
  video_bytes="$(stat -c '%s' "${video}")"
  if (( video_bytes >= 98000000 )); then
    printf 'published video exceeds safe GitHub size: %s (%s bytes)\n' "${video}" "${video_bytes}" >&2
    exit 3
  fi
done < <(find "${PUBLISH_DIR}" -maxdepth 1 -type f -name '*.mp4' -print0)

if [[ "${PUBLISH_RESULTS}" == "1" ]]; then
  rm -rf "${PUBLISH_DIR}/diagnostics"
  git add "${PUBLISH_DIR}"
  if ! git diff --cached --quiet; then
    git -c user.name="${GIT_RESULT_AUTHOR_NAME}" \
      -c user.email="${GIT_RESULT_AUTHOR_EMAIL}" \
      commit -m "results: add canonical MANO extended visual reviews"
  fi
  git push origin HEAD:main
fi

write_status completed "exit_code=0" "stage=published" "current_dataset=all"
FINALIZED=1
printf '[extended-mano] completed and published: %s\n' "${PUBLISH_DIR}"
