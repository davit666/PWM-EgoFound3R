#!/usr/bin/env python3
"""Two optimizer steps on a real scene clip; not an accuracy experiment."""
import json
from pathlib import Path
import sys
recipe = json.loads(Path('configs/training_reconstruction_example.json').read_text())
recipe.update(steps=2,batch_size=1,save_every=1,warmup_steps=1,
              stride_probabilities=[1.,0.,0.,0.,0.])
Path(sys.argv[1]).write_text(json.dumps(recipe,indent=2))
