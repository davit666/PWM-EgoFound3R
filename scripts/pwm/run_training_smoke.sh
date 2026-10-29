#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BASE=/home-mixed/liwenhao666/wm/PWM-EgoFound3R
PYTHON_BIN="$BASE/.venv/bin/python"
KIND="${1:-stera}"
case "$KIND" in stera|taco) ;; *) echo 'expected stera or taco'; exit 2;; esac
SUFFIX="${KIND/taco/_taco}"
[[ "$KIND" != stera ]] || SUFFIX=''
RUN_ROOT="/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke${SUFFIX}_20261029"
RESULT_DIR="docs/results/training_smoke${SUFFIX}_20261029"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
export PYTHONPATH="$PWD/outputs/training-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT" "$RESULT_DIR"
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT_DIR/status.json"
  for file in preparation.json train.log eval.json label_projection.png; do
    [[ ! -f "$RUN_ROOT/$file" ]] || cp "$RUN_ROOT/$file" "$RESULT_DIR/"
  done
  for file in metrics.jsonl status.json; do
    [[ ! -f "$RUN_ROOT/run/$file" ]] || cp "$RUN_ROOT/run/$file" "$RESULT_DIR/training_$file"
  done
  git add -- "$RESULT_DIR"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: real STERA training and checkpoint smoke'
  fi
  git push origin "HEAD:refs/heads/pwm/training-smoke${SUFFIX}"
  exit "$code"
}
trap publish EXIT
exec > >(tee "$RUN_ROOT/train.log") 2>&1
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-7]$ ]] || { echo 'smoke requires exactly one GPU'; exit 2; }
used=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 100 )) || { echo 'requested GPU is busy'; exit 2; }
[[ ! -e "$RUN_ROOT/run/latest.pt" ]] || { echo 'smoke already has checkpoint; use a new explicit run path'; exit 2; }
[[ -e checkpoints ]] || ln -s "$BASE/checkpoints" checkpoints
if [[ "$KIND" == stera ]]; then
  "$PYTHON_BIN" scripts/pwm/prepare_stera_scene.py \
    --session /share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905 \
    --output "$RUN_ROOT"
else
  [[ -d outputs/TACO-Instructions/.git ]] || git clone https://github.com/leolyliu/TACO-Instructions.git outputs/TACO-Instructions
  git -C outputs/TACO-Instructions checkout --detach fd3b5a6b39565ca6935a91d35d5e91dde03c5142
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps --no-deps chumpy
  "$PYTHON_BIN" scripts/pwm/prepare_taco_hands.py \
    --root /share-mixed/project/eai_pwm/datasets/TACO --taco-code outputs/TACO-Instructions \
    --mano-root /share-mixed/project/eai_pwm/datasets/ARCTIC/body_models/mano --output "$RUN_ROOT"
fi
"$PYTHON_BIN" scripts/pwm/make_training_smoke_recipe.py "$RUN_ROOT/recipe.json"
common=(--config configs/pwm_baseline.toml --recipe "$RUN_ROOT/recipe.json" --index "$RUN_ROOT/index.jsonl")
"$PYTHON_BIN" scripts/pwm/train_prepared.py "${common[@]}" --output "$RUN_ROOT/run" --preflight-only
"$PYTHON_BIN" scripts/pwm/train_prepared.py "${common[@]}" --output "$RUN_ROOT/run" --initialize "$BASE/checkpoints/EgoFound3R.pt" --stop-after 1
"$PYTHON_BIN" scripts/pwm/train_prepared.py "${common[@]}" --output "$RUN_ROOT/run" --resume "$RUN_ROOT/run/latest.pt"
"$PYTHON_BIN" scripts/pwm/evaluate_prepared.py "${common[@]}" --checkpoint "$RUN_ROOT/run/latest.pt" --split train --output "$RUN_ROOT/eval.json"
