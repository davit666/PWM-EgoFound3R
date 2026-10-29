#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN=/home-mixed/liwenhao666/wm/PWM-EgoFound3R/.venv/bin/python
RUN=/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs/training_smoke_taco_20261029
RESULT=docs/results/trained_checkpoint_inference_20261029
mkdir -p "$RESULT"
exec 3>&1 4>&2
export CUDA_VISIBLE_DEVICES=0
publish() {
  code=$?
  trap - EXIT
  exec 1>&3 2>&4
  [[ -z "${LOG_TEE_PID:-}" ]] || wait "$LOG_TEE_PID"
  printf '{"exit_code":%s}\n' "$code" > "$RESULT/status.json"
  for file in summary.json overlay.mp4; do
    [[ ! -f "$RUN/inference/$file" ]] || cp "$RUN/inference/$file" "$RESULT/"
  done
  git add -- "$RESULT"
  if ! git diff --cached --quiet; then
    git -c user.name='Wenhao Li' -c user.email='wl2655@columbia.edu' commit -m 'results: trained checkpoint full video inference export'
  fi
  git push origin HEAD:refs/heads/pwm/trained-checkpoint-inference
  exit "$code"
}
trap publish EXIT
exec > >(tee "$RESULT/run.log") 2>&1
LOG_TEE_PID=$!
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 100 )) || { echo 'GPU0 is busy'; exit 2; }
"$PYTHON_BIN" scripts/pwm/prepared_clip_video.py "$RUN/taco_hands.npz" "$RUN/prepared.mp4"
"$PYTHON_BIN" infer_marker_video.py --config configs/pwm_baseline.toml \
  --checkpoint "$RUN/run/latest.pt" --video-path "$RUN/prepared.mp4" --output-dir "$RUN/inference" \
  --device cuda:0 --input-height 256 --input-width 256 --high-fps 30 --global-stride 5 \
  --upsample-mano --overlay-marker-count 778 --no-overlay-frame-images
"$PYTHON_BIN" scripts/pwm/validate_output.py "$RUN/inference" > "$RESULT/validation.json"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RESULT/final_gpu_processes.csv"
