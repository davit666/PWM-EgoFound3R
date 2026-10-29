#!/usr/bin/env python3
"""Bounded host probe; never reads credential contents or modifies mounts."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

report={'host':os.uname().nodename,'python':sys.executable,'modules':{}}
for name in ('torch','torchvision','numpy','cv2','scipy','einops','safetensors','huggingface_hub','h5py','trimesh','deepspeed','chumpy','boto3'):
    try:
        spec=importlib.util.find_spec(name)
        report['modules'][name]={'origin':spec.origin if spec else None}
        try: report['modules'][name]['version']=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    except Exception as e: report['modules'][name]={'error':str(e)}
report['tools']={name:shutil.which(name) for name in ('python','python3','uv','git','ffmpeg','ks3','rclone','nvidia-smi','tmux')}
report['paths']={str(p):{'exists':p.exists(),'directory':p.is_dir()} for p in map(Path,[
 '/share/project/ks3util/ks3util-linux-amd64','/share/project/ks3util/ks3util-config',
 '/share-mixed/project/ks3util/ks3util-linux-amd64','/share-mixed/project/ks3util/ks3util-config',
 '/home/liwenhao666/.cache/huggingface/token','/root/.cache/huggingface/token',
 '/share-mixed/projset/eai_pwm/datasets/TACO','/share-mixed/projset/eai_pwm/datasets/Stera-10M',
 '/share-mixed/projset/eai_pwm/datasets/ARCTIC','/home/liwenhao666/.venv/bin/python'])}
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
 '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
report['disk']=subprocess.run(['df','-B1','/share/project','/share-mixed/projset'],capture_output=True,text=True).stdout
report['dataset_names']=sorted(p.name for p in Path('/share-mixed/projset/eai_pwm/datasets').iterdir())
report['home_names']={str(p):sorted(x.name for x in p.iterdir()) for p in
 [Path('/share-mixed/projset/eai_pwm/home'),Path('/home/liwenhao666')] if p.is_dir()}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
