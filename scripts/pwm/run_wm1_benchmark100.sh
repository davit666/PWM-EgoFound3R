#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$PWD" == /share-mixed/projset/eai_pwm/experiments/PWM-benchmark100 ]] || { echo 'wm1 workspace required'; exit 2; }
MODE=${1:?expected prepare-stera, prepare-taco, train, resume or evaluate}
RUN="$PWD/outputs/benchmark100_taco_stera_20261029"
RESULT=docs/results/wm1_benchmark100_20261029
mkdir -p "$RUN" "$RESULT" "$RUN/tmp" "$RUN/extensions"
export PYTHONPATH="$PWD/outputs/training-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$RUN/tmp" TORCH_EXTENSIONS_DIR="$RUN/extensions"
export PWM_DISTRIBUTED_TIMEOUT_SECONDS=180
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export DS_BUILD_OPS=0
MONITOR_PID=''; GPU_PID=''; PHASE_OK=0
exec 3>&1 4>&2
publish() { python3 scripts/pwm/publish_benchmark_progress.py "$RUN" "$RESULT" "$1"; }
stop_monitors() {
  for pid in "$MONITOR_PID" "$GPU_PID"; do
    [[ -z "$pid" ]] || { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
  done
  MONITOR_PID=''; GPU_PID=''
}
finish() {
  code=$?
  trap - EXIT
  if (( code==0 && PHASE_OK==0 )); then code=130; fi
  stop_monitors
  exec 1>&3 2>&4
  [[ -z "${LOG_TEE_PID:-}" ]] || wait "$LOG_TEE_PID" || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN/final_gpu_processes.csv"
  printf '{"exit_code":%s,"mode":"%s"}\n' "$code" "$MODE" > "$RESULT/job_status.json"
  phase="$MODE-completed"
  (( code==0 )) || phase="$MODE-failed"
  publish "$phase" || { echo 'result publication failed'; (( code!=0 )) || code=1; }
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$RUN/run.log") 2>&1
LOG_TEE_PID=$!
for gpu in {0..7}; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used<100 )) || { echo "GPU $gpu is busy"; exit 2; }
done
df -B1 /share-mixed/projset > "$RUN/storage_before.txt"
df -hT /share-mixed/projset >> "$RUN/storage_before.txt"
if [[ -d /share/project ]]; then
  df -B1 /share/project >> "$RUN/storage_before.txt"
else
  echo '/share/project absent; no writes possible there' >> "$RUN/storage_before.txt"
fi
publish "$MODE-starting"
(
  while sleep 60; do
    df -B1 /share-mixed/projset >> "$RUN/storage_watchdog.log"
    [[ ! -d /share/project ]] || df -B1 /share/project >> "$RUN/storage_watchdog.log"
    python3 scripts/pwm/publish_benchmark_progress.py "$RUN" "$RESULT" "$MODE" >> "$RUN/publish.log" 2>&1 || true
  done
) &
MONITOR_PID=$!
case "$MODE" in
  stage)
    export HF_HOME="$PWD/outputs/hf-cache" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
    export XDG_CACHE_HOME="$PWD/outputs/cache"
    python3 scripts/pwm/stage_wm1_benchmark_data.py
    ;;
  prepare-stera|prepare-taco)
    dataset=STERA-10M; subdir=stera
    [[ "$MODE" != prepare-taco ]] || { dataset=TACO; subdir=taco; }
    python3 scripts/pwm/prepare_benchmark_subset.py --output "$RUN/$subdir" --datasets "$dataset" \
      --taco-root "$PWD/outputs/data/TACO" --stera-root /share-mixed/projset/eai_pwm/datasets/Stera-10M \
      --taco-code "$PWD/outputs/TACO-Instructions" --mano-root "$PWD/outputs/data/mano" --frames 60 --taco-workers 8
    ;;
  train|resume)
    mkdir -p "$RUN/data"
    python3 scripts/pwm/combine_prepared_indices.py "$RUN/data/index.jsonl" "$RUN/taco/index.jsonl" "$RUN/stera/index.jsonl"
    python3 scripts/pwm/preflight_benchmark_data.py --index "$RUN/data/index.jsonl" --output "$RUN/data/preflight.json" \
      --frames 60 --datasets TACO STERA-10M --eval-datasets TACO STERA-10M
    [[ -f "$RUN/recipe.json" ]] || cp configs/training_reconstruction_example.json "$RUN/recipe.json"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop=10 > "$RUN/gpu.csv" &
    GPU_PID=$!
    common=(--config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" --index "$RUN/data/index.jsonl" --output "$RUN/run")
    if [[ "$MODE" == train ]]; then
      [[ ! -e "$RUN/run/sharded/latest" ]] || { echo 'existing checkpoint requires resume'; exit 2; }
      # Keep the full 3200-step schedule. The first update also validates 60-frame
      # memory use; it counts toward the requested 100 updates.
      python3 -m torch.distributed.run --standalone --nproc_per_node=8 scripts/pwm/train_zero2.py "${common[@]}" --save-initial --stop-after 1
      cp "$RUN/run/timing.json" "$RUN/run/timing_step1.json"
    fi
    python3 -m torch.distributed.run --standalone --nproc_per_node=8 scripts/pwm/train_zero2.py "${common[@]}" --resume --stop-after 100
    ;;
  evaluate)
    for tag in initial 100; do
      checkpoint="$RUN/run/initial.pt"
      [[ "$tag" != 100 ]] || checkpoint="$RUN/run/inference.pt"
      python3 scripts/pwm/evaluate_prepared.py --config configs/pwm_baseline.toml --recipe "$RUN/recipe.json" \
        --index "$RUN/data/index.jsonl" --checkpoint "$checkpoint" --split val --stride 5 --output "$RUN/eval_$tag.json"
    done
    python3 scripts/pwm/report_benchmark100.py "$RUN"
    ;;
  *) echo 'unknown phase'; exit 2 ;;
esac
PHASE_OK=1
