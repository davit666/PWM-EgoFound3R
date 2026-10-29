#!/usr/bin/env python3
"""Single-device reconstruction runner; explicit experimental recipe required.

The released inference architecture is reused without topology changes. This is
not an assertion of the unpublished authors' dataloader or training recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from egohandmetric_prompt.builders import build_runtime_marker_model
from egohandmetric_prompt.checkpoints import (
    load_marker_model_weights, MARKER_CHECKPOINT_NAMESPACE,
    MARKER_CHECKPOINT_SCHEMA_VERSION, ROOT_GEOMETRY_RUNTIME_CONTRACT,
)
from egohandmetric_prompt.configs import (
    load_project_config, marker_architecture_descriptor, marker_architecture_hash,
)
from egohandmetric_prompt.multirate_contracts import build_multirate_frame_map
from egohandmetric_prompt.training.data import PreparedClips
from egohandmetric_prompt.training.objectives import objective
from egohandmetric_prompt.training.scene import gather_scene_targets


def native_training_outputs(result):
    # Select H explicitly so no interpolated R tensor becomes a training target.
    output = result.high_rate.to_dict()
    output.update(depth_global=result.global_3r.depth, depth_conf_global=result.global_3r.depth_conf,
                  camera_pose_global=result.global_3r.camera_pose,
                  intrinsics_global=result.global_3r.intrinsics,
                  log_metric_value=result.metric.log_metric_value)
    return output


def learning_rate(step, recipe):
    warmup, steps = recipe["warmup_steps"], recipe["steps"]
    if step < warmup:
        return recipe["lr"] * (step + 1) / warmup
    phase = (step - warmup) / max(steps - warmup - 1, 1)
    return recipe["min_lr"] + (recipe["lr"] - recipe["min_lr"]) * (1 + math.cos(math.pi * phase)) / 2


def save_checkpoint(path, model, optimizer, step, config, recipe, index_hash):
    payload = {
        "checkpoint_schema_version": MARKER_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_namespace": MARKER_CHECKPOINT_NAMESPACE,
        "temporal_architecture": marker_architecture_descriptor(config.marker_model),
        "temporal_architecture_hash": marker_architecture_hash(config.marker_model),
        "root_geometry_contract": ROOT_GEOMETRY_RUNTIME_CONTRACT,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
        "optimizer": optimizer.state_dict(), "step": step, "recipe": recipe,
        "index_sha256": index_hash, "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Stop at this optimizer step without changing the LR recipe")
    args = parser.parse_args()
    if args.initialize and args.resume:
        parser.error("choose initialization or exact resume")
    recipe = json.loads(args.recipe.read_text())
    required = {"steps", "batch_size", "lr", "min_lr", "warmup_steps", "weight_decay",
                "confidence_alpha", "loss_weights", "stride_probabilities", "seed",
                "save_every", "precision", "assumptions_acknowledged"}
    if required - set(recipe) or not recipe.get("assumptions_acknowledged"):
        raise ValueError("explicit recipe and acknowledged reconstruction assumptions required")
    probabilities = recipe["stride_probabilities"]
    if len(probabilities) != 5 or any(p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1):
        raise ValueError("stride probabilities must sum to 1 over strides 1..5")
    if min(recipe[k] for k in ("steps", "batch_size", "save_every")) < 1:
        raise ValueError("steps, batch size and save interval must be positive")
    if recipe["precision"] not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    dataset = PreparedClips(args.index, "train")
    index_hash = hashlib.sha256(args.index.read_bytes()).hexdigest()
    first = dataset[0]
    print(json.dumps({"clips": len(dataset), "input_shape": list(first["images"].shape),
                      "targets": sorted(first["targets"]), "index_sha256": index_hash}))
    if args.preflight_only:
        for i in range(len(dataset)):
            dataset[i]
        print(json.dumps({'state': 'validated', 'clips_checked': len(dataset)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "latest.pt").exists() and not args.resume:
        raise FileExistsError("existing run requires --resume or a new output directory")
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    config = load_project_config(args.config)
    model = build_runtime_marker_model(config)
    initial = args.resume or args.initialize
    if initial:
        # Published checkpoint dtype may be bf16. Load in that dtype, then keep
        # FP32 optimizer master parameters for training with autocast.
        load_marker_model_weights(model, initial, model_config=config.marker_model,
                                  align_model_floating_dtype=True)
    model = model.float().to(args.device).train()
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("backbone must remain frozen")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=recipe["lr"], weight_decay=recipe["weight_decay"],
                                 betas=(0.9, 0.999), eps=1e-8)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["recipe"] != recipe or saved["index_sha256"] != index_hash:
            raise ValueError("resume recipe/index mismatch")
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        random.setstate(saved["rng_python"])
        torch.set_rng_state(saved["rng_torch"])
        if saved["rng_cuda"]:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
    try:
        with (args.output / "metrics.jsonl").open("a") as log:
            end = min(recipe['steps'], args.stop_after or recipe['steps'])
            if end <= start:
                raise ValueError('requested stop step must exceed resumed step')
            for step in range(start, end):
                optimizer.zero_grad(set_to_none=True)
                rate = learning_rate(step, recipe)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                reports = []
                # Accumulate one independently masked clip at a time: mixed
                # datasets need neither common target keys nor image shapes.
                for _ in range(recipe["batch_size"]):
                    sample = dataset[random.randrange(len(dataset))]
                    images = sample["images"][None].to(args.device)
                    targets = {k: (v[None].to(args.device), m[None].to(args.device))
                               for k, (v, m) in sample["targets"].items()}
                    present = torch.ones(images.shape[:2], dtype=torch.bool, device=args.device)
                    stride = random.choices(range(1, 6), weights=probabilities)[0]
                    mapping = build_multirate_frame_map(high_frame_present=present,
                        high_temporal_pair_valid=present[:, :-1] & present[:, 1:],
                        global_stride=stride, global_anchor_phase=min(2, stride - 1))
                    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                        enabled=recipe["precision"] == "bfloat16"):
                        output = native_training_outputs(model(images, multirate_frame_map=mapping))
                        total, terms = objective(output, gather_scene_targets(targets, mapping), recipe["loss_weights"],
                                                 confidence_alpha=recipe["confidence_alpha"],
                                                 heatmap_positive_weight=recipe.get('heatmap_positive_weight', 1.))
                    (total / recipe["batch_size"]).backward()
                    reports.append({k: float(v.detach()) for k, v in terms.items()})
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                record = {"step": step + 1, "lr": rate, "gradient_norm": float(norm), "terms": reports}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if (step + 1) % recipe["save_every"] == 0 or step + 1 == end:
                    save_checkpoint(args.output / "latest.pt", model, optimizer, step + 1,
                                    config, recipe, index_hash)
        (args.output / "status.json").write_text(json.dumps({"state": "completed" if end == recipe['steps'] else 'stopped_at_requested_step', "steps": end}))
    except Exception as error:
        (args.output / "status.json").write_text(json.dumps({"state": "failed", "error": str(error)}))
        raise


if __name__ == "__main__":
    main()
