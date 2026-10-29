#!/usr/bin/env python3
"""Two optimizer steps on a real scene clip; not an accuracy experiment."""
import json
from pathlib import Path
import sys
recipe = json.loads(Path('configs/training_reconstruction_example.json').read_text())
recipe.update(steps=2,batch_size=1,save_every=1,warmup_steps=1,
              stride_probabilities=[1.,0.,0.,0.,0.])
if len(sys.argv)>2:
    stride=int(sys.argv[2])
    if stride not in range(1,6): raise ValueError('stride must be 1..5')
    recipe['stride_probabilities']=[float(i==stride) for i in range(1,6)]
if len(sys.argv)>3:
    recipe['batch_size']=int(sys.argv[3])
    if recipe['batch_size']<1: raise ValueError('accumulation must be positive')
Path(sys.argv[1]).write_text(json.dumps(recipe,indent=2))
