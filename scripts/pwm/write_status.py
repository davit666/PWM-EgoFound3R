#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("state", choices=("running", "completed", "failed"))
    parser.add_argument("--video")
    parser.add_argument("--output-dir")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    value = {
        "schema_version": "pwm.egofound3r.run_status.v1",
        "state": args.state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "video": args.video,
        "output_dir": args.output_dir,
        "exit_code": args.exit_code,
    }
    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
