#!/usr/bin/env python3
"""Export the same held-out RGB windows with initial and step-100 checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

run=Path(sys.argv[1]).resolve()
destination=run/'previews'
destination.mkdir(exist_ok=True)
rows=[json.loads(line) for line in (run/'data/index.jsonl').read_text().splitlines()]
manifest=[]
for dataset in ('TACO','STERA-10M'):
    row=next(row for row in rows if row['dataset']==dataset and row['split']=='val')
    clip=Path(row['path'])
    video=destination/f'{dataset}_input.mp4'
    subprocess.run([sys.executable,'scripts/pwm/prepared_clip_video.py',str(clip),str(video)],check=True)
    with np.load(clip,allow_pickle=False) as arrays:
        fps=round(1/float(np.median(np.diff(arrays['timestamp']))),6)
    outputs=[]
    for tag,filename in [('initial','initial.pt'),('step100','inference.pt')]:
        output=destination/f'{dataset}_{tag}'
        subprocess.run([sys.executable,'infer_marker_video.py','--config','configs/pwm_baseline.toml',
            '--checkpoint',str(run/'run'/filename),'--video-path',str(video),'--output-dir',str(output),
            '--device','cuda:0','--input-height','256','--input-width','256','--input-fps',str(fps),'--high-fps',str(fps),
            '--global-stride','5','--upsample-mano','--overlay-marker-count','778','--no-overlay-frame-images','--allow-output-overwrite'],check=True)
        subprocess.run([sys.executable,'scripts/pwm/validate_output.py',str(output)],check=True)
        outputs.append(output/'overlay.mp4')
    target=destination/f'{dataset}_initial_vs_step100.mp4'
    filters="[0:v]pad=iw:ih+32:0:32:black,drawtext=text='Initial':x=12:y=6:fontsize=20:fontcolor=white[a];[1:v]pad=iw:ih+32:0:32:black,drawtext=text='Step 100':x=12:y=6:fontsize=20:fontcolor=white[b];[a][b]hstack=inputs=2[v]"
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(outputs[0]),'-i',str(outputs[1]),
        '-filter_complex',filters,'-map','[v]','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(target)],check=True)
    manifest.append({'dataset':dataset,'identity':row,'comparison':target.name,
        'left':'released builder initialization with pretrained frozen VGGT-Omega','right':'100 updates',
        'scope':'full inference preview; quantitative validation uses the exact prepared tensors'})
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
