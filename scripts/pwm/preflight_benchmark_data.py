#!/usr/bin/env python3
"""Fail closed before timing: verify dataset coverage and disjoint real clips.

This checks our prepared data contract. It cannot certify equality with the
unreleased author preprocessing or original train/test split.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from egohandmetric_prompt.training.data import PreparedClips

PAPER_DATASETS={'H2O','HOT3D-Aria','OakInk-v2','TACO','HOI4D','ARCTIC','STERA-10M'}


def inspect_index(index, expected_datasets, frames, eval_datasets=None):
    if frames<2:
        raise ValueError('a temporal benchmark needs at least two frames')
    hashes={}
    report={}
    eval_datasets=set(expected_datasets)-{'STERA-10M'} if eval_datasets is None else set(eval_datasets)
    for split in ('train','val'):
        data=PreparedClips(index,split)
        counts=defaultdict(int)
        episodes=defaultdict(set)
        targets=defaultdict(set)
        for sample in data:
            row=sample['identity']
            if not row.get('sha256') or not row.get('adapter'):
                raise ValueError('every clip needs a checksum and versioned adapter')
            if row['sha256'] in hashes:
                raise ValueError('duplicate prepared clip, including across split boundaries')
            hashes[row['sha256']]=(split,row['dataset'],row['source_episode_id'])
            if tuple(sample['images'].shape)!=(frames,3,256,256):
                raise ValueError(f"unexpected training input shape: {row['path']}")
            if not sample['targets'] or not any(mask.any() for _,mask in sample['targets'].values()):
                raise ValueError('clip has no active supervision')
            counts[row['dataset']]+=1
            episodes[row['dataset']].add(row['source_episode_id'])
            targets[row['dataset']].update(sample['targets'])
        expected=set(expected_datasets) if split=='train' else eval_datasets
        if set(counts)!=expected:
            raise ValueError(f'{split} dataset mismatch: expected {sorted(expected)}, got {sorted(counts)}')
        # Engineering benchmark minimum, not a claim about the authors' recipe.
        if split=='train' and min(counts.values())<8:
            raise ValueError('at least eight distinct train clips per dataset required for this benchmark')
        report[split]={name:{'clips':counts[name],'source_episodes':sorted(episodes[name]),
            'targets':sorted(targets[name])} for name in sorted(counts)}
    report.update(frames_per_clip=frames,image_hw=[256,256],
        scope='prepared-contract check; original author preprocessing and splits are not certified')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frames',required=True,type=int,help='explicit reconstruction choice: training clip length was not disclosed')
    parser.add_argument('--datasets',nargs='+',default=sorted(PAPER_DATASETS))
    parser.add_argument('--eval-datasets',nargs='+',help='defaults to train datasets excluding STERA, matching paper evaluation coverage')
    args=parser.parse_args()
    result=inspect_index(args.index,set(args.datasets),args.frames,args.eval_datasets)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
