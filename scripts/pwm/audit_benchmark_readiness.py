#!/usr/bin/env python3
"""Read-only current-host inventory for the requested seven-dataset benchmark."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_training_sources import ROOTS

report={'host':os.uname().nodename,'datasets':{},'project':str(Path.cwd())}
for name,original in ROOTS.items():
    suffix=original.split('/eai_pwm/',1)[-1]
    candidates=[Path(root)/'eai_pwm'/suffix for root in
        ('/share/project','/share/projset','/share-mixed/project','/share-mixed/projset')]
    if name=='HOI4D':
        candidates += [Path(root)/'eai_pwm'/child for root in
            ('/share/project','/share/projset','/share-mixed/project','/share-mixed/projset')
            for child in ('datasets/HOI4D','datasets/hoi4d','datasets/raw/HOI4D','raw/data/HOI4D')]
        candidates += [Path(root)/child for root in
            ('/share/project','/share/projset','/share-mixed/project','/share-mixed/projset')
            for child in ('dataset/ego/HOI4D','dataset/HOI4D')]
    report['datasets'][name]={'candidates':{str(p):p.is_dir() for p in candidates}}
    for path in candidates:
        if path.is_dir():
            with os.scandir(path) as entries:
                sample=[]
                for entry in entries:
                    sample.append({'name':entry.name,'directory':entry.is_dir(),'bytes':entry.stat().st_size})
                    if len(sample)>=20: break
            report['datasets'][name].setdefault('samples',{})[str(path)]=sample
base=Path('/share-mixed/project/eai_pwm/datasets')
hot=base/'Hot3DAria'
if hot.is_dir():
    sequences=[]
    for path in sorted(hot.iterdir()):
        if not path.is_dir(): continue
        files=[{'name':p.name,'bytes':p.stat().st_size} for p in path.iterdir() if p.is_file()]
        sequences.append({'sequence':path.name,
            'video_files':[x for x in files if Path(x['name']).suffix.lower() in ('.vrs','.mp4','.avi','.mkv')],
            'mano_bytes':next((x['bytes'] for x in files if x['name']=='mano_hand_pose_trajectory.jsonl'),0),
            'file_names':[x['name'] for x in files] if len(sequences)<3 else []})
        if len(sequences)>=256: break
    report['hot3d_sequence_modalities']=sequences
report['annotation_directories']={}
for path in (base/'ARCTIC/downloads/data',base/'OakInk-v2/anno_preview'):
    if path.is_dir():
        report['annotation_directories'][str(path)]=[p.name for p in sorted(path.iterdir())[:30]]
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
    '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
report['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
    '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
report['paths']={}
for path in ('/home-mixed/liwenhao666/wm/PWM-EgoFound3R/.venv/bin/python',
    '/home-mixed/liwenhao666/wm/PWM-training-reconstruction',
    '/share-mixed/project/eai_pwm/PWM-EgoFound3R/outputs',
    '/share/project/eai_pwm/PWM-EgoFound3R/outputs'):
    p=Path(path)
    report['paths'][path]={'exists':p.exists(),'writable':os.access(p,os.W_OK)}
    if p.is_dir(): report['paths'][path]['free_bytes']=shutil.disk_usage(p).free
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
