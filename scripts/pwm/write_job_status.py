#!/usr/bin/env python3
"""Atomically merge fields into a small machine-readable job status."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _coerce(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("state")
    parser.add_argument("fields", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    data: dict[str, object] = {}
    if args.path.is_file():
        data = json.loads(args.path.read_text(encoding="utf-8"))
    data.update(
        schema_version="pwm.job_status.v1",
        state=args.state,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    for field in args.fields:
        if "=" not in field:
            raise ValueError(f"expected KEY=VALUE, got {field!r}")
        key, value = field.split("=", 1)
        data[key] = _coerce(value)

    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.path.with_name(f".{args.path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.path)


if __name__ == "__main__":
    main()
