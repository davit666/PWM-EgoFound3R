#!/usr/bin/env python3
"""Archive actual timing and paired validation; never infer completion from age."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from egohandmetric_prompt.training.timing import summarize_step_times

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args()
run=args.run
ranks=[[json.loads(line) for line in (run/f'run/rank{i}.jsonl').read_text().splitlines()] for i in range(8)]
for i,records in enumerate(ranks):
    if [r['step'] for r in records]!=list(range(1,101)):
        raise ValueError(f'rank {i} did not complete exactly steps 1..100')
    for row in records:
        if row['world_size']!=8 or len(row['sources'])!=8 or not math.isfinite(row['gradient_norm']):
            raise ValueError('invalid world size, accumulation or gradients')
        if not all(math.isfinite(v) for term in row['terms'] for v in term.values()):
            raise ValueError('nonfinite training loss')
before=json.loads((run/'eval_initial.json').read_text())
after=json.loads((run/'eval_100.json').read_text())
if [w['identity'] for w in before['windows']] != [w['identity'] for w in after['windows']]:
    raise ValueError('before/after validation cohort changed')
comparison={}
for dataset,metrics in before['by_dataset'].items():
    comparison[dataset]={}
    for name,first in metrics.items():
        last=after['by_dataset'][dataset][name]
        comparison[dataset][name]={'initial':first['mean'],'step100':last['mean'],
            'delta':last['mean']-first['mean'] if first['mean'] is not None and last['mean'] is not None else None,
            'initial_windows':first['windows'],'final_windows':last['windows']}
report={'ok':True,'scope':'100-step TACO/STERA pilot; not original seven-dataset paper reproduction',
    'timing':summarize_step_times(ranks[0]),'world_size':8,'effective_batch_size':64,
    'frames_per_clip':sorted({n for row in ranks[0] for n in row['frames_per_clip']}),
    'peak_allocated_gib':max(r['peak_memory_allocated_bytes'] for rows in ranks for r in rows)/2**30,
    'unique_training_clips_seen':len({k for rows in ranks for r in rows for k in r['clip_ids']}),
    'validation_windows':len(before['windows']),'comparison':comparison}
(run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='comparison'}))
