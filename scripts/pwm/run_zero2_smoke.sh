#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN=/home-mixed/liwenhao666/wm/PWM-EgoFound3R/.venv/bin/python
BASE=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke_taco_20261029
MODE=${1:-two}
case "$MODE" in
  two) DEVICES=0,1; WORLD=2; ACCUMULATION=1; NAME=training_zero2_smoke_20261029; BRANCH=pwm/zero2-smoke ;;
  eight) DEVICES=0,1,2,3,4,5,6,7; WORLD=8; ACCUMULATION=8; NAME=training_zero2_batch64_20261029; BRANCH=pwm/zero2-batch64 ;;
  *) echo 'usage: run_zero2_smoke.sh [two|eight]'; exit 2 ;;
esac
RUN=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/$NAME
RESULT=docs/results/$NAME
mkdir -p "$RUN" "$RESULT"
mkdir -p "$RESULT/attempts"
[[ ! -s "$RESULT/run.log" ]] || cp "$RESULT/run.log" "$RESULT/attempts/$(date +%Y%m%dT%H%M%S).log"
exec 3>&1 4>&2
export CUDA_VISIBLE_DEVICES=$DEVICES
export PYTHONPATH="$PWD/outputs/training-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DS_BUILD_OPS=0
export OMP_NUM_THREADS=4
export PWM_DISTRIBUTED_TIMEOUT_SECONDS=120
export TORCH_DISTRIBUTED_DEBUG=DETAIL
publish() {
  code=$?
  trap - EXIT
  exec 1>&3 2>&4
  [[ -z "${LOG_TEE_PID:-}" ]] || wait "$LOG_TEE_PID" || true
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/status.json"
  for source in "$RUN"/run/rank*.jsonl; do
    [[ ! -f "$source" ]] || cp "$source" "$RESULT/training_$(basename "$source")"
  done
  for file in deepspeed_config.json status.json; do
    [[ ! -f "$RUN/run/$file" ]] || cp "$RUN/run/$file" "$RESULT/training_$file"
  done
  [[ ! -f "$RUN/eval.json" ]] || cp "$RUN/eval.json" "$RESULT/"
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: two-GPU ZeRO-2 training and resume smoke'
  fi
  git push origin "HEAD:refs/heads/$BRANCH"
  exit "$code"
}
trap publish EXIT
exec > >(tee "$RESULT/run.log") 2>&1
LOG_TEE_PID=$!
IFS=, read -ra gpu_ids <<< "$DEVICES"
for gpu in "${gpu_ids[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 100 )) || { echo "GPU${gpu} is busy"; exit 2; }
done
[[ ! -e "$RUN/run/sharded/latest" ]] || { echo 'existing run; explicitly resume instead'; exit 2; }
"$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps --no-deps --no-build-isolation deepspeed==0.19.8
"$PYTHON_BIN" -m pip install --disable-pip-version-check --target outputs/training-deps pydantic psutil ninja hjson py-cpuinfo msgpack
"$PYTHON_BIN" scripts/pwm/make_training_smoke_recipe.py "$RUN/recipe.json" 5 "$ACCUMULATION"
"$PYTHON_BIN" scripts/pwm/combine_prepared_indices.py "$RUN/index.jsonl" "$BASE/index.jsonl" \
  /share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke_20261029/index.jsonl
common=(--config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" --index "$RUN/index.jsonl" --output "$RUN/run")
timeout --kill-after=30s 8m "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" scripts/pwm/train_zero2.py "${common[@]}" \
  --initialize /home-mixed/liwenhao666/wm/PWM-EgoFound3R/checkpoints/EgoFound3R.pt --stop-after 1
timeout --kill-after=30s 8m "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" scripts/pwm/train_zero2.py "${common[@]}" --resume
"$PYTHON_BIN" scripts/pwm/evaluate_prepared.py --config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" \
  --index "$RUN/index.jsonl" --checkpoint "$RUN/run/inference.pt" --split train --output "$RUN/eval.json"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RESULT/final_gpu_processes.csv"
