#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

KS3UTIL="${KS3UTIL:-/share/project/ks3util/ks3util-linux-amd64}"
KS3_CONFIG="${KS3_CONFIG:-/share/project/ks3util/ks3util-config}"
SOURCE_URI="${SOURCE_URI:-ks3://baai-eai/ours-ego-sampled/}"
DEST_ROOT="${DEST_ROOT:-/share-mixed/projset/eai_pwm/datasets/ours-ego-sampled}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/ours_ego_sampled_transfer}"
STATUS_PATH="${STATUS_PATH:-${RUN_ROOT}/status.json}"
TRANSFER_LOG="${TRANSFER_LOG:-${RUN_ROOT}/transfer.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${RUN_ROOT}/storage-watchdog.log}"
EXPECTED_OBJECTS="${EXPECTED_OBJECTS:-5502}"
EXPECTED_BYTES="${EXPECTED_BYTES:-121242302679}"
PROJECT_WARN_AVAILABLE_BYTES="${PROJECT_WARN_AVAILABLE_BYTES:-43980465111040}"
PROJECT_CRITICAL_AVAILABLE_BYTES="${PROJECT_CRITICAL_AVAILABLE_BYTES:-32985348833280}"
PROJECT_GROWTH_LIMIT_BYTES="${PROJECT_GROWTH_LIMIT_BYTES:-10737418240}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-300}"
FINALIZED=0
WATCHDOG_PID=""

write_status() {
  python3 scripts/pwm/write_job_status.py "${STATUS_PATH}" "$@"
}

filesystem_used_bytes() {
  df -B1 --output=used "$1" | awk 'NR == 2 {print $1}'
}

filesystem_available_bytes() {
  df -B1 --output=avail "$1" | awk 'NR == 2 {print $1}'
}

stop_watchdog() {
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
  fi
}

on_exit() {
  local exit_code=$?
  stop_watchdog
  if [[ "${FINALIZED}" -eq 0 ]]; then
    write_status failed "exit_code=${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT INT TERM

[[ -x "${KS3UTIL}" ]] || { printf 'missing ks3util: %s\n' "${KS3UTIL}" >&2; exit 2; }
[[ -r "${KS3_CONFIG}" ]] || { printf 'missing ks3 config: %s\n' "${KS3_CONFIG}" >&2; exit 2; }
case "${DEST_ROOT}" in
  /share/mixed/projset/*|/share-mixed/projset/*) ;;
  *) printf 'refusing non-mixed destination: %s\n' "${DEST_ROOT}" >&2; exit 2 ;;
esac
case "${RUN_ROOT}" in
  /share/mixed/projset/*|/share-mixed/projset/*) ;;
  *) printf 'refusing non-mixed run root: %s\n' "${RUN_ROOT}" >&2; exit 2 ;;
esac

endpoint="$(awk -F= '$1 == "endpoint" {print $2}' "${KS3_CONFIG}")"
proxy="$(awk -F= '$1 == "proxy" {print $2}' "${KS3_CONFIG}")"
[[ "${endpoint}" == "ks3-cn-beijing-internal.ksyuncs.com" ]] || {
  printf 'unexpected KS3 endpoint: %s\n' "${endpoint}" >&2
  exit 2
}
[[ "${proxy}" == "http://10.8.36.21:2080" ]] || {
  printf 'unexpected KS3 proxy: %s\n' "${proxy}" >&2
  exit 2
}

mkdir -p "${DEST_ROOT}" "${RUN_ROOT}" "${RUN_ROOT}/ks3util-logs"
project_used_before="$(filesystem_used_bytes /share/project)"
project_available_before="$(filesystem_available_bytes /share/project)"
mixed_available_before="$(filesystem_available_bytes /share/mixed/projset)"
df -B1 /share/project /share/mixed/projset > "${RUN_ROOT}/df-before-bytes.txt"
df -hT /share/project /share/mixed/projset > "${RUN_ROOT}/df-before-human.txt"

if (( project_available_before < PROJECT_CRITICAL_AVAILABLE_BYTES )); then
  printf 'warning: /share/project is already below the critical free-space threshold; it remains read-only for this job\n' >&2
elif (( project_available_before < PROJECT_WARN_AVAILABLE_BYTES )); then
  printf 'warning: /share/project is below the free-space warning threshold\n' >&2
fi
if (( mixed_available_before < EXPECTED_BYTES * 2 )); then
  printf 'insufficient mixed free space for download plus working headroom\n' >&2
  exit 2
fi

write_status preflight \
  "source_uri=${SOURCE_URI}" "destination=${DEST_ROOT}" \
  "expected_objects=${EXPECTED_OBJECTS}" "expected_bytes=${EXPECTED_BYTES}" \
  "project_used_before=${project_used_before}" \
  "project_available_before=${project_available_before}" \
  "mixed_available_before=${mixed_available_before}"

(
  while true; do
    now="$(date --iso-8601=seconds)"
    project_used="$(filesystem_used_bytes /share/project)"
    project_available="$(filesystem_available_bytes /share/project)"
    mixed_available="$(filesystem_available_bytes /share/mixed/projset)"
    project_growth=$((project_used - project_used_before))
    printf '%s project_used=%s project_available=%s project_growth=%s mixed_available=%s\n' \
      "${now}" "${project_used}" "${project_available}" "${project_growth}" "${mixed_available}" \
      >> "${WATCHDOG_LOG}"
    if (( project_available < PROJECT_CRITICAL_AVAILABLE_BYTES || project_growth > PROJECT_GROWTH_LIMIT_BYTES )); then
      printf '%s ALERT project storage threshold crossed\n' "${now}" >> "${WATCHDOG_LOG}"
    fi
    sleep "${WATCHDOG_INTERVAL_SECONDS}"
  done
) &
WATCHDOG_PID=$!

write_status transferring "watchdog_pid=${WATCHDOG_PID}"
printf '[transfer] %s -> %s\n' "${SOURCE_URI}" "${DEST_ROOT}"
"${KS3UTIL}" -c "${KS3_CONFIG}" \
  sync -u -f --jobs=32 --parallel=2 \
  "${SOURCE_URI}" "${DEST_ROOT}/" --read-timeout 6 \
  --log-path="${RUN_ROOT}/ks3util-logs" --loglevel=info \
  2>&1 | tee "${TRANSFER_LOG}"

write_status verifying
read -r local_objects local_bytes < <(
  find "${DEST_ROOT}" -type f -printf '%s\n' \
    | awk '{count += 1; bytes += $1} END {printf "%d %.0f\n", count, bytes}'
)
project_used_after="$(filesystem_used_bytes /share/project)"
project_available_after="$(filesystem_available_bytes /share/project)"
mixed_available_after="$(filesystem_available_bytes /share/mixed/projset)"
df -B1 /share/project /share/mixed/projset > "${RUN_ROOT}/df-after-bytes.txt"
df -hT /share/project /share/mixed/projset > "${RUN_ROOT}/df-after-human.txt"

if [[ "${local_objects}" != "${EXPECTED_OBJECTS}" || "${local_bytes}" != "${EXPECTED_BYTES}" ]]; then
  write_status failed \
    "exit_code=3" "local_objects=${local_objects}" "local_bytes=${local_bytes}" \
    "project_used_after=${project_used_after}" "mixed_available_after=${mixed_available_after}"
  printf 'verification mismatch: objects=%s bytes=%s\n' "${local_objects}" "${local_bytes}" >&2
  exit 3
fi

project_growth=$((project_used_after - project_used_before))
write_status completed \
  "exit_code=0" "local_objects=${local_objects}" "local_bytes=${local_bytes}" \
  "project_used_after=${project_used_after}" \
  "project_available_after=${project_available_after}" \
  "project_growth_bytes=${project_growth}" \
  "mixed_available_after=${mixed_available_after}"
FINALIZED=1
stop_watchdog
printf '[transfer] completed and verified: objects=%s bytes=%s\n' "${local_objects}" "${local_bytes}"
