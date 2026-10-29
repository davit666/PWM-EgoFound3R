#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
case "$PWD" in /share-mixed/projset/eai_pwm/experiments/PWM-benchmark100) ;; *) echo 'unexpected wm1 workspace'; exit 2;; esac
mkdir -p outputs/bootstrap outputs/training-deps outputs/tools outputs/secrets
chmod 700 outputs/secrets
export PIP_CACHE_DIR="$PWD/outputs/bootstrap/pip-cache"
export TMPDIR="$PWD/outputs/bootstrap"
export DS_BUILD_OPS=0
RESULT=docs/results/wm1_benchmark100_20261029
mkdir -p "$RESULT"
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/bootstrap_status.json"
  cp outputs/bootstrap/bootstrap.log "$RESULT/bootstrap.log"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: bootstrap isolated wm1 benchmark dependencies'
  fi
  git push origin HEAD:refs/heads/pwm/wm1-benchmark100
  exit "$code"
}
trap publish EXIT
exec > outputs/bootstrap/bootstrap.log 2>&1
python3 -m pip install --disable-pip-version-check --target outputs/training-deps --no-deps --no-build-isolation \
  deepspeed==0.19.8 h5py trimesh chumpy
python3 -m pip install --disable-pip-version-check --target outputs/training-deps pydantic psutil ninja hjson py-cpuinfo msgpack
[[ -e checkpoints ]] || ln -s /share-mixed/projset/eai_pwm/home/liwenhao/wm/PWM-EgoFound3R/checkpoints checkpoints
[[ -d outputs/TACO-Instructions/.git ]] || git clone https://github.com/leolyliu/TACO-Instructions.git outputs/TACO-Instructions
git -C outputs/TACO-Instructions checkout --detach fd3b5a6b39565ca6935a91d35d5e91dde03c5142
if [[ ! -x outputs/tools/ks3util ]]; then
  curl -fL --retry 2 https://ks3util-version-update.ks3-cn-beijing.ksyuncs.com/2.15.0/ks3util-linux-amd64 -o outputs/tools/ks3util
  chmod 755 outputs/tools/ks3util
fi
PYTHONPATH="$PWD/outputs/training-deps:$PWD" python3 -c 'import torch, deepspeed, h5py, trimesh; print("IMPORTS_OK",torch.__version__,deepspeed.__version__)'
