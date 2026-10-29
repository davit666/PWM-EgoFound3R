#!/usr/bin/env python3
"""Verify every rank and archive bounded evidence from a completed smoke run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


def verify(run, archive):
    status=json.loads((run/'run/status.json').read_text())
    recipe=json.loads((run/'recipe.json').read_text())
    config=json.loads((run/'run/deepspeed_config.json').read_text())
    world=status['world_size']
    if status['state']!='completed' or status['steps']!=recipe['steps']:
        raise ValueError('training did not complete')
    if config['train_batch_size']!=world*recipe['batch_size']:
        raise ValueError('effective batch mismatch')
    ranks=[]
    for rank in range(world):
        path=run/f'run/rank{rank}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if [row['step'] for row in rows]!=list(range(1,recipe['steps']+1)):
            raise ValueError(f'rank {rank} has incomplete/duplicate updates')
        for row in rows:
            if row['rank']!=rank or row['world_size']!=world:
                raise ValueError('rank identity mismatch')
            if len(row['sources'])!=recipe['batch_size'] or len(row['terms'])!=recipe['batch_size']:
                raise ValueError('gradient accumulation did not cover the requested clips')
            values=[v for term in row['terms'] for v in term.values()]
            if not all(math.isfinite(v) for v in values+[row['gradient_norm']]) or row['gradient_norm']<=0:
                raise ValueError('nonfinite loss or missing gradients')
        ranks.append(rows)
        shutil.copy2(path,archive/f'training_rank{rank}.jsonl')
    for step in zip(*ranks):
        if any(not math.isclose(row['gradient_norm'],step[0]['gradient_norm'],rel_tol=1e-6) for row in step):
            raise ValueError('ranks disagree on the reduced gradient norm')
    evaluation=json.loads((run/'eval.json').read_text())
    if evaluation['loaded_state_keys']<=0 or len(evaluation['windows'])!=2:
        raise ValueError('portable export did not pass both prepared clips')
    if len((archive/'final_gpu_processes.csv').read_text().strip().splitlines())!=1:
        raise ValueError('final GPU process inventory is not empty')
    digest=hashlib.sha256()
    with (run/'run/inference.pt').open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    result={'ok':True,'world_size':world,'optimizer_steps':status['steps'],
        'clips_per_rank_per_step':recipe['batch_size'],'effective_batch_size':config['train_batch_size'],
        'gradient_norms':[row['gradient_norm'] for row in ranks[0]],
        'portable_checkpoint_sha256':digest.hexdigest(),'loaded_state_keys':evaluation['loaded_state_keys'],
        'evaluation_scope':'two training clips, 3 frames each; engineering validation only',
        'gpu_processes_released':True}
    (archive/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    verify(args.run,args.archive)
