#!/usr/bin/env python3
"""Plot measured update times and losses from the archived eight-rank run."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args()
ranks=[[json.loads(line) for line in (args.archive/f'training_rank{i}.jsonl').read_text().splitlines()] for i in range(8)]
recipe=json.loads((args.archive/'recipe.json').read_text())
steps=[r['step'] for r in ranks[0]]
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
axes[0,0].plot(steps,[r['step_seconds'] for r in ranks[0]],color='#277da8')
axes[0,0].set(title='Optimizer update time',ylabel='Seconds')
for dataset,color in [('TACO','#d45b45'),('STERA-10M','#26966f')]:
    values=[]
    for index in range(len(steps)):
        losses=[sum(recipe['loss_weights'][key]*value for key,value in terms.items())
            for rank in ranks for source,terms in zip(rank[index]['sources'],rank[index]['terms'])
            if source.split('/')[0]==dataset]
        values.append(np.mean(losses) if losses else np.nan)
    axes[0,1].plot(steps,values,label=dataset,color=color,alpha=.8)
axes[0,1].set(title='Weighted loss by dataset (all ranks)',ylabel='Loss')
axes[0,1].legend(frameon=False)
axes[1,0].plot(steps,[r['gradient_norm'] for r in ranks[0]],color='#9256a6')
axes[1,0].set(title='Global gradient norm before clipping',ylabel='Norm',yscale='log')
axes[1,1].plot(steps,[max(rank[i]['peak_memory_allocated_bytes'] for rank in ranks)/2**30 for i in range(len(steps))],color='#d18b27')
axes[1,1].set(title='Peak allocated memory across eight GPUs',ylabel='GiB')
for axis in axes.flat:
    axis.set_xlabel('Optimizer step')
    axis.grid(alpha=.2)
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('PWM / EgoFound3R — TACO + STERA pilot, 60 frames, effective batch 64')
fig.savefig(args.archive/'training_curves.png',dpi=160)
