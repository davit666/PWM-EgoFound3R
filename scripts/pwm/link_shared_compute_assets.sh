#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${1:-/share-mixed/projset/eai_pwm/home/liwenhao/wm/PWM-EgoFound3R}"

case "${REPO_ROOT}" in
  /home-mixed/liwenhao666/*|/share-mixed/project/*) ;;
  *) printf 'refusing non-persistent compute checkout: %s\n' "${REPO_ROOT}" >&2; exit 2 ;;
esac
case "${SOURCE_ROOT}" in
  /share-mixed/projset/*) ;;
  *) printf 'refusing unexpected shared asset source: %s\n' "${SOURCE_ROOT}" >&2; exit 2 ;;
esac

link_asset() {
  local source="$1"
  local destination="$2"
  [[ -e "${source}" ]] || {
    printf 'missing shared asset: %s\n' "${source}" >&2
    exit 2
  }
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ "$(readlink -f "${destination}")" == "$(readlink -f "${source}")" ]] || {
      printf 'refusing to replace existing destination: %s\n' "${destination}" >&2
      exit 2
    }
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  ln -s "${source}" "${destination}"
}

link_asset "${SOURCE_ROOT}/.venv" "${REPO_ROOT}/.venv"
link_asset "${SOURCE_ROOT}/checkpoints/EgoFound3R.pt" \
  "${REPO_ROOT}/checkpoints/EgoFound3R.pt"
link_asset "${SOURCE_ROOT}/checkpoints/vggt_omega_1b_512.pt" \
  "${REPO_ROOT}/checkpoints/vggt_omega_1b_512.pt"
link_asset "${SOURCE_ROOT}/other/mano_neutral" "${REPO_ROOT}/other/mano_neutral"

"${REPO_ROOT}/.venv/bin/python" -c \
  'import av, cv2, mcap, numpy, torch; print("python-assets=ok")'
printf 'linked shared compute assets from %s\n' "${SOURCE_ROOT}"
