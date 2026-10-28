#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KS3UTIL="${KS3UTIL:-/share/project/ks3util/ks3util-linux-amd64}"
KS3_CONFIG="${KS3_CONFIG:-/share/project/ks3util/ks3util-config}"
KS3_PREFIX="${KS3_PREFIX:-ks3://baai-eai/project-assets/PWM-EgoFound3R/mano-neutral/}"
DEST_DIR="${DEST_DIR:-${REPO_ROOT}/other/mano_neutral}"
LEFT_SHA256="1d2d1ceb73b8ba647df7afa6f76daac3cdad7c49259a997c6f3b95c19dc4a342"
RIGHT_SHA256="ea2e2524cb7ad83246b87c0d6ba9e5b294cfaf6b639abaa853cd3fbc28f3eacb"

case "${DEST_DIR}" in
  /share/mixed/projset/*|/share-mixed/projset/*) ;;
  *) printf 'refusing non-mixed MANO asset destination: %s\n' "${DEST_DIR}" >&2; exit 2 ;;
esac

for path in "${KS3UTIL}" "${KS3_CONFIG}"; do
  [[ -r "${path}" ]] || {
    printf 'missing required KS3 path: %s\n' "${path}" >&2
    exit 2
  }
done

printf '[neutral-mano] storage preflight\n'
df -B1 /share/project /share/mixed/projset
df -hT /share/project /share/mixed/projset
mkdir -p "${DEST_DIR}"

"${KS3UTIL}" -c "${KS3_CONFIG}" sync -u -f --jobs=4 --parallel=2 \
  "${KS3_PREFIX}" "${DEST_DIR}/" --read-timeout 6

printf '%s  %s\n' "${LEFT_SHA256}" "${DEST_DIR}/mano_left_neutral.npz" \
  | sha256sum -c -
printf '%s  %s\n' "${RIGHT_SHA256}" "${DEST_DIR}/mano_right_neutral.npz" \
  | sha256sum -c -
stat -c '%n %s bytes' \
  "${DEST_DIR}/mano_left_neutral.npz" \
  "${DEST_DIR}/mano_right_neutral.npz"
printf '[neutral-mano] staged from %s to %s\n' "${KS3_PREFIX}" "${DEST_DIR}"
