#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from model_assets import ASSETS, ModelAsset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, asset: ModelAsset) -> dict[str, object]:
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != asset.size_bytes:
        raise RuntimeError(
            f"size mismatch for {path}: expected {asset.size_bytes}, got {size}"
        )
    if digest != asset.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {asset.sha256}, got {digest}"
        )
    return {
        "repo_id": asset.repo_id,
        "filename": asset.filename,
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": digest,
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and checksum the two gated PWM-EgoFound3R checkpoints."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            LocalTokenNotFoundError,
        )
    except ModuleNotFoundError:
        print(
            "huggingface_hub is not installed; run scripts/pwm/bootstrap_env.sh first",
            file=sys.stderr,
        )
        return 3
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verified: list[dict[str, object]] = []
    try:
        for asset in ASSETS:
            destination = output_dir / asset.filename
            if not destination.is_file():
                print(f"Downloading {asset.repo_id}/{asset.filename} ...", flush=True)
                downloaded = Path(
                    hf_hub_download(
                        repo_id=asset.repo_id,
                        filename=asset.filename,
                        local_dir=output_dir,
                        token=True,
                    )
                )
                if downloaded.resolve() != destination.resolve():
                    raise RuntimeError(
                        f"unexpected download path: {downloaded} (expected {destination})"
                    )
            print(f"Verifying {destination} ...", flush=True)
            verified.append(verify(destination, asset))
    except (GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError) as error:
        print(
            "Checkpoint access was denied. Log in with `hf auth login` and accept "
            "the terms for both gated model pages:",
            file=sys.stderr,
        )
        for asset in ASSETS:
            print(f"  {asset.terms_url}", file=sys.stderr)
        print(f"Hugging Face response: {type(error).__name__}", file=sys.stderr)
        return 2

    manifest = {
        "schema_version": "pwm.egofound3r.checkpoints.v1",
        "assets": verified,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
