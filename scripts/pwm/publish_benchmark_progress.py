#!/usr/bin/env python3
"""Small Git heartbeat snapshots; partial last JSONL lines are ignored."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
run,result=map(Path,sys.argv[1:3])
phase=sys.argv[3]
result.mkdir(parents=True,exist_ok=True)
rows=[]
path=run/'run/rank0.jsonl'
if path.exists():
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
step=rows[-1]['step'] if rows else 0
state={'phase':phase,'step':step,'target_steps':100,'world_size':8,'effective_batch_size':64}
if rows:
    state.update(last_step_seconds=rows[-1]['step_seconds'],last_gradient_norm=rows[-1]['gradient_norm'])
progress=result/'progress.json'
if progress.exists() and json.loads(progress.read_text())==state and phase=='training':
    sys.exit(0)
progress.write_text(json.dumps(state,indent=2)+'\n')
for name in ('recipe.json','data/preflight.json','data/summary.json','run/timing.json','run/timing_step1.json',
    'run/initialization.json','run/deepspeed_config.json','run/status.json','eval_initial.json','eval_100.json','report.json'):
    source=run/name
    if source.is_file(): shutil.copy2(source,result/name.replace('/','_'))
if phase not in ('preparing','training'):
    for source in (run/'run').glob('rank*.jsonl'): shutil.copy2(source,result/('training_'+source.name))
    for name in ('gpu.csv','final_gpu_processes.csv'):
        if (run/name).exists(): shutil.copy2(run/name,result/name)
if (run/'run.log').exists():
    with (run/'run.log').open('rb') as file:
        file.seek(max(0,(run/'run.log').stat().st_size-100000))
        (result/'run_tail.log').write_bytes(file.read())
subprocess.run(['git','add','--',str(result)],check=True)
changed=subprocess.run(['git','diff','--cached','--quiet']).returncode
if changed:
    subprocess.run(['git','-c','user.name=Wenhao Li','-c','user.email=wl2655@columbia.edu','commit','-m',f'results: wm1 benchmark {phase} at step {step}'],check=True)
subprocess.run(['git','push','origin','HEAD:refs/heads/pwm/wm1-benchmark100'],check=True)
