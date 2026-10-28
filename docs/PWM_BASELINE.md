# PWM-EgoFound3R baseline reproduction

## Scope

This repository is a PWM research fork of the official EgoFound3R inference
release at upstream commit `264f4df9fd42536aae4dba982d18272cc3334077`.
Upstream currently publishes pretrained inference code, not the training or
evaluation pipeline used to produce all paper tables. Therefore baseline v0 is
defined as a provenance-recorded, checksum-verified inference run whose output
contract is automatically validated.

The internal Python model names remain `egofound3r` because they are part of
the checkpoint topology contract. PWM branding is applied at the repository,
configuration and run-workflow layers.

## Frozen model assets

| Asset | Expected bytes | SHA-256 | Source |
| --- | ---: | --- | --- |
| `EgoFound3R.pt` | 423,526,586 | `02ba3135b030a23f88b91a986c173f6a3c600f6da45637605fd09914e9d78e66` | <https://huggingface.co/Frank-F2021/EgoFound3R> |
| `vggt_omega_1b_512.pt` | 4,576,706,117 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` | <https://huggingface.co/facebook/VGGT-Omega> |

Both repositories are gated. Accept their terms with the same Hugging Face
account used by `hf auth login`, then run:

```bash
.venv/bin/python scripts/pwm/download_checkpoints.py
```

The downloader refuses wrong sizes or hashes and writes
`checkpoints/manifest.json`.

## Cuihu v0 data input

The first smoke run uses 20 frames from a locally stored P0 STERA-10M RGB
video:

```text
/share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905/rgb.mp4
```

This only establishes end-to-end execution. Dataset-level reproduction and
metrics are separate milestones.

## Run

```bash
PYTHON_BIN=python3 bash scripts/pwm/bootstrap_env.sh
hf auth login
.venv/bin/python scripts/pwm/download_checkpoints.py

CUDA_VISIBLE_DEVICES=0 \
ALLOW_OUTPUT_OVERWRITE=1 \
bash scripts/pwm/run_baseline.sh \
  /share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905/rgb.mp4
```

The workflow writes `summary.json`, `inference_output.pt`, overlays,
`validation.json`, and an adjacent machine-readable run status. A run is only
complete when `validate_output.py` succeeds.

## Cuihu durable launch

Launch through an already-connected outer Cuihu tmux and keep the actual job
in a remote inner tmux:

```bash
cd /share-mixed/projset/eai_pwm/home/liwenhao/wm/PWM-EgoFound3R
tmux new-session -d -s pwm-egofound3r-v0 \
  "CUDA_VISIBLE_DEVICES=0 ALLOW_OUTPUT_OVERWRITE=1 bash scripts/pwm/run_baseline.sh \
  /share-mixed/projset/eai_pwm/datasets/Stera-10M/session_data_20260330_014905/rgb.mp4 \
  > outputs/pwm_baseline_v0.log 2>&1"
```
