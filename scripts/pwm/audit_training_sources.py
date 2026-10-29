#!/usr/bin/env python3
"""Bounded, read-only inventory; directory existence is not label readiness."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOTS = {
    'H2O': '/share/project/eai_pwm/datasets/H2O',
    'HOT3D-Aria': '/share/project/eai_pwm/datasets/Hot3DAria',
    'OakInk-v2': '/share/project/eai_pwm/datasets/OakInk-v2',
    'TACO': '/share/project/eai_pwm/datasets/TACO',
    'HOI4D': '/share/project/eai_pwm/datasets/raw/data/HOI4D',
    'ARCTIC': '/share/project/eai_pwm/datasets/ARCTIC',
    'STERA-10M': '/share-mixed/projset/eai_pwm/datasets/Stera-10M',
}

def inventory(root):
    root = Path(root)
    result = {'root': str(root), 'exists': root.is_dir(), 'samples': [], 'bounded': True}
    if not root.is_dir():
        return result
    queue = [(root, 0)]
    start = time.monotonic()
    while queue and len(result['samples']) < 160 and time.monotonic() - start < 45:
        directory, depth = queue.pop(0)
        try:
            with os.scandir(directory) as stream:
                entries = []
                for entry in stream:
                    entries.append(entry)
                    if len(entries) >= 60:
                        break
            for entry in entries:
                path = Path(entry.path)
                is_dir = entry.is_dir(follow_symlinks=True)
                item = {'path': str(path.relative_to(root)), 'directory': is_dir}
                if not is_dir:
                    item['bytes'] = entry.stat().st_size
                result['samples'].append(item)
                if is_dir and depth < 5:
                    queue.append((path, depth + 1))
                if len(result['samples']) >= 160:
                    break
        except OSError as error:
            result.setdefault('errors', []).append(str(error))
    return result

def main():
    report = {'sources': {k: inventory(v) for k,v in ROOTS.items()},
              'gpu': subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                                     '--format=csv,noheader'], capture_output=True, text=True).stdout,
              'host': os.uname().nodename}
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {'exists': v['exists'], 'samples': len(v['samples'])} for k,v in report['sources'].items()}))

if __name__ == '__main__':
    main()
