#!/usr/bin/env python3
"""Combine prepared manifests without moving or rewriting their data files."""
import json
from pathlib import Path
import sys
rows=[]
for filename in sys.argv[2:]:
    source=Path(filename).resolve()
    for line in source.read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        row['path']=str((source.parent/row['path']).resolve())
        rows.append(row)
Path(sys.argv[1]).write_text(''.join(json.dumps(r)+'\n' for r in rows))
