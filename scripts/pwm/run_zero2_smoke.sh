#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN=/home-mixed/liwenhao666/wm/PWM-EgoFound3R/.venv/bin/python
BASE=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke_taco_20261029
RUN=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_zero2_smoke_20261029
RESULT=docs/results/training_zero2_smoke_20261029
mkdir -p "$RUN" "$RESULT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/outputs/training-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DS_BUILD_OPS=0
export OMP_NUM_THREADS=4
publish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/status.json"
  for file in rank0.jsonl rank1.jsonl deepspeed_config.json status.json; do
    [[ ! -f "$RUN/run/$file" ]] || cp "$RUN/run/$file" "$RESULT/training_$file"
  done
  [[ ! -f "$RUN/eval.json" ]] || cp "$RUN/eval.json" "$RESULT/"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: two-GPU ZeRO-2 training and resume smoke'
  fi
  git push origin HEAD:refs/heads/pwm/zero2-smoke
  exit "$code"
}
trap publish EXIT
exec > >(tee "$RESULT/run.log") 2>&1
for gpu in 0 1; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 100 )) || { echo "GPU${gpu} is busy"; exit 2; }
done
[[ ! -e "$RUN/run/sharded/latest" ]] || { echo 'existing run; explicitly resume instead'; exit 2; }
"$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps --no-deps --no-build-isolation deepspeed
"$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps pydantic psutil ninja hjson py-cpuinfo msgpack
"$PYTHON_BIN" scripts/pwm/make_training_smoke_recipe.py "$RUN/recipe.json"
common=(--config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" --index "$BASE/index.jsonl" --output "$RUN/run")
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/pwm/train_zero2.py "${common[@]}" \
  --initialize /home-mixed/liwenhao666/wm/PWM-EgoFound3R/checkpoints/EgoFound3R.pt --stop-after 1
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/pwm/train_zero2.py "${common[@]}" --resume
"$PYTHON_BIN" scripts/pwm/evaluate_prepared.py --config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" \
  --index "$BASE/index.jsonl" --checkpoint "$RUN/run/inference.pt" --split train --output "$RUN/eval.json"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RESULT/final_gpu_processes.csv"
