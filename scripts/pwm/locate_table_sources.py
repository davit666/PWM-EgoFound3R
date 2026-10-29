#!/usr/bin/env python3
"""Bounded read-only search around the spreadsheet's current/legacy mounts."""
import json
import os
from pathlib import Path
import sys
import time

roots=[Path(root)/child for root in ('/share/project','/share/projset','/share-mixed/project','/share-mixed/projset')
       for child in ('eai_pwm/datasets','dataset/ego','dataset/tactile')]
report={'spreadsheet':'/home/zy/Downloads/EgoFound3R 数据集推荐清单.xlsx',
    'table_paths':['/share/project/eai_pwm/datasets/raw/data/HOI4D','/share/project/eai_pwm/datasets/Hot3DAria'],
    'roots':{str(p):p.is_dir() for p in roots},'hits':[],'errors':[],
    'scope':'directory names, archive names and VRS/MP4 files near table paths; no content extraction'}
start=time.monotonic(); seen=set(); scanned=0; queue=[(p,0) for p in roots if p.is_dir()]
while queue and scanned<50000 and time.monotonic()-start<90:
    path,depth=queue.pop(0)
    try:
        resolved=str(path.resolve())
        if resolved in seen: continue
        seen.add(resolved)
        with os.scandir(path) as stream:
            for entry in stream:
                scanned+=1
                lower=entry.name.lower()
                p=Path(entry.path)
                isdir=entry.is_dir()
                relevant='hoi4d' in lower or 'hot3d' in lower or p.suffix.lower()=='.vrs'
                if relevant:
                    report['hits'].append({'path':str(p),'directory':isdir,'bytes':entry.stat().st_size})
                if not isdir and p.suffix.lower() in ('.mp4','.zip','.tar','.gz') and any(x in str(path).lower() for x in ('hot3d','hoi4d')):
                    report['hits'].append({'path':str(p),'directory':False,'bytes':entry.stat().st_size})
                if isdir and depth<6 and (depth>0 or any(x in lower for x in ('raw','hot','hoi','ego','download','archive'))):
                    queue.append((p,depth+1))
                if scanned>=50000 or time.monotonic()-start>=90: break
    except OSError as error:
        if len(report['errors'])<20: report['errors'].append(str(error))
report.update(entries_scanned=scanned,directories_scanned=len(seen),elapsed_seconds=time.monotonic()-start,
              bounded_search_incomplete=bool(queue))
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
