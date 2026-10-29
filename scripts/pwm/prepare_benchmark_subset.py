#!/usr/bin/env python3
"""Prepare a bounded TACO/STERA pilot with source-episode-disjoint validation.

Uses original TACO train/test membership; STERA holdout is our own pilot split.
The 60-frame window is a declared choice, not an unpublished author training rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def split_prepared(directory, destination, frames):
    row=json.loads((directory/'index.jsonl').read_text().strip())
    result=[]
    with np.load(directory/row['path'],allow_pickle=False) as data:
        arrays={name:data[name] for name in data.files}
        count=len(arrays['rgb'])
        if count%frames: raise ValueError('whole windows required')
        for number,start in enumerate(range(0,count,frames)):
            target=destination/f'clip_{number:02d}.npz'
            payload={name:value[start:start+frames] if value.ndim and len(value)==count else value for name,value in arrays.items()}
            np.savez_compressed(target,**payload)
            child={**row,'path':str(target.resolve()),'source_frames':row['source_frames'][start:start+frames],
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'preparation_report':str((directory/'preparation.json').resolve())}
            result.append(child)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('taco-root','stera-root','taco-code','mano-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--frames',type=int,default=60)
    parser.add_argument('--taco-train-sequences',type=int,default=16)
    parser.add_argument('--taco-val-sequences',type=int,default=4)
    parser.add_argument('--taco-workers',type=int,default=1)
    parser.add_argument('--datasets',nargs='+',choices=['TACO','STERA-10M'],default=['TACO','STERA-10M'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parent
    split_map=dict(line.strip().split(',') for line in (args.taco_code/'data_lists/v1_overall_data_train_test_split.txt').read_text().splitlines() if line.strip())
    candidates=[]
    for line in (args.taco_code/'data_lists/v1_egocentric_data_available_sequences.txt').read_text().splitlines():
        if line.strip():
            triplet,sequence=line.rsplit(' ',1)
            candidates.append((triplet,sequence))
    # First use different activities, then further sequences. Never change a
    # dataset test sequence to train just because an earlier conversion failed.
    by_group={}
    for triplet,sequence in candidates:
        by_group.setdefault(triplet,[]).append(sequence)
    candidates=[(triplet,sequences[i]) for i in range(max(map(len,by_group.values())))
                for triplet,sequences in sorted(by_group.items()) if i<len(sequences)]
    rows=[]; attempts=[]
    available={}
    if 'TACO' in args.datasets:
        with zipfile.ZipFile(args.taco_root/'Egocentric_RGB_Videos.zip') as archive:
            available={x.filename:x.file_size for x in archive.infolist()}
    attempt_lock=threading.Lock()
    def execute(command,directory,gpu=None):
        directory.mkdir(parents=True,exist_ok=True)
        reused=False
        if (directory/'index.jsonl').is_file() and (directory/'preparation.json').is_file():
            row=json.loads((directory/'index.jsonl').read_text())
            wanted=int(command[command.index('--frames')+1])
            start=int(command[command.index('--start')+1])
            path=directory/row['path']
            reused=(row['source_frames']==list(range(start,start+wanted)) and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'])
        code=0
        if not reused:
            env=os.environ.copy()
            if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (directory/'adapter.log').open('w') as log:
                code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
        with attempt_lock:
            attempts.append({'command':list(map(str,command)),'returncode':code,'reused':reused,'log':str(directory/'adapter.log')})
            temporary=args.output/'attempts.tmp'
            temporary.write_text(json.dumps(attempts,indent=2))
            temporary.replace(args.output/'attempts.json')
        return code==0
    for split,needed in [('train',args.taco_train_sequences),('val',args.taco_val_sequences)]:
        if 'TACO' not in args.datasets: break
        count=0; jobs=[]
        for triplet,sequence in candidates:
            native=split_map[sequence]
            if (split=='train')!=(native=='train'): continue
            member=f'Egocentric_RGB_Videos/{triplet}/{sequence}/color.mp4'
            if member not in available or available[member]>256*1024*1024: continue
            if len(jobs)>=needed*3: break
            directory=args.output/f'taco_{split}_{sequence}'
            length=args.frames*(2 if split=='train' else 1)
            command=[sys.executable,str(script/'prepare_taco_hands.py'),'--root',str(args.taco_root),
                '--taco-code',str(args.taco_code),'--mano-root',str(args.mano_root),'--output',str(directory),
                '--triplet',triplet,'--sequence',sequence,'--start','30','--frames',str(length),'--split',split]
            jobs.append((sequence,native,directory,command))
        if not 1<=args.taco_workers<=8: raise ValueError('one to eight TACO workers supported')
        with ThreadPoolExecutor(max_workers=args.taco_workers) as pool:
            for offset in range(0,len(jobs),args.taco_workers):
                batch=jobs[offset:offset+args.taco_workers]
                futures=[pool.submit(execute,command,directory,gpu) for gpu,(_,_,directory,command) in enumerate(batch)]
                for (sequence,native,directory,_),future in zip(batch,futures):
                    if not future.result():
                        print(json.dumps({'skipped':sequence,'log':str(directory/'adapter.log')}),flush=True)
                        continue
                    if count>=needed: continue
                    child=split_prepared(directory,directory,args.frames)
                    for row in child: row['dataset_native_split']=native
                    rows.extend(child);count+=1
                    print(json.dumps({'dataset':'TACO','split':split,'prepared_sequences':count,'clips':len(rows)}),flush=True)
                if count==needed: break
        if count!=needed: raise RuntimeError(f'not enough valid TACO {split} sequences: {count}/{needed}')
    sessions=[p for p in sorted(args.stera_root.glob('session*')) if (p/'rgb.mp4').is_file() and (p/'annotation.hdf5').is_file()]
    used=set()
    for split,needed,clips_per_session in [('train',4,8),('val',2,2)]:
        if 'STERA-10M' not in args.datasets: break
        count=0; tried=0
        for session in sessions:
            if session.name in used: continue
            tried+=1
            if tried>needed*3: break
            accepted=[]
            for attempt in range(clips_per_session+8):
                directory=args.output/f'stera_{split}_{session.name}_{attempt:02d}'
                command=[sys.executable,str(script/'prepare_stera_scene.py'),'--session',str(session),
                    '--output',str(directory),'--start',str(31+attempt*args.frames),'--frames',str(args.frames),'--split',split]
                if execute(command,directory): accepted.extend(split_prepared(directory,directory,args.frames))
                if len(accepted)==clips_per_session: break
            # Any inspected session is kept out of the other split, including
            # failed or partially accepted sessions.
            used.add(session.name)
            if len(accepted)!=clips_per_session: continue
            rows.extend(accepted);count+=1
            print(json.dumps({'dataset':'STERA-10M','split':split,'prepared_sessions':count,'clips':len(rows)}),flush=True)
            if count==needed: break
        if count!=needed: raise RuntimeError(f'not enough valid STERA {split} sessions: {count}/{needed}')
    (args.output/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report={'rows':len(rows),'frames_per_clip':args.frames,'scope':'TACO/STERA pilot; not original seven-dataset training split',
        'native_taco_split':'train versus test_1/test_2','stera_split':'our source-session holdout',
        'counts':{f'{dataset}/{split}':sum(r['dataset']==dataset and r['split']==split for r in rows)
                  for dataset in ('TACO','STERA-10M') for split in ('train','val')}}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
