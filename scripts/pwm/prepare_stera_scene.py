#!/usr/bin/env python3
"""Small real-label STERA scene sample. Hand labels are deliberately not inferred.

Supports the audited sparse-detection release with frame_timestamps. Optical->
link calibration is applied before inverting camera-link->world poses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--start', type=int, default=31)
    parser.add_argument('--split',choices=['train','val'],default='train')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cal = args.session / 'calibrations'
    meta = json.loads((cal/'meta.json').read_text())
    K = np.load(cal/'rgb_K.npy').astype(np.float64)
    D = np.load(cal/'rgb_D.npy')
    DK, DD = np.load(cal/'depth_K.npy'), np.load(cal/'depth_D.npy')
    rotation = np.load(cal/'R_optical_to_link.npy')
    if rotation.shape != (3,3) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4):
        raise ValueError('invalid optical-to-link calibration')
    transform = np.array([[256/meta['rgb']['width'],0,(256/meta['rgb']['width']-1)/2],
                          [0,256/meta['rgb']['height'],(256/meta['rgb']['height']-1)/2],[0,0,1]])
    new_K = transform @ K
    # Both streams are aligned optical depth; remap to common rays/intrinsics.
    rgb_map = cv2.initUndistortRectifyMap(K,D if D.size else None,None,new_K,(256,256),cv2.CV_32FC1)
    dep_map = cv2.initUndistortRectifyMap(DK,DD if DD.size else None,None,new_K,(256,256),cv2.CV_32FC1)
    frames = np.arange(args.start,args.start+args.frames)
    cap = cv2.VideoCapture(str(args.session/'rgb.mp4'))
    # Sequential decoding preserves exact frame indexing; do not timestamp seek.
    rgb = []
    for i in range(int(frames[-1])+1):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'video decode failed at frame {i}')
        if i >= args.start:
            rgb.append(cv2.cvtColor(cv2.remap(image,*rgb_map,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB))
    cap.release()
    with h5py.File(args.session/'annotation.hdf5','r') as src:
        time = src['hand-pose/frame_timestamps'][frames]
        pose_times = src['cam-pose/timestamps'][:]
        nearest = np.abs(pose_times[:,None]-time[None]).argmin(0)
        if np.max(np.abs(pose_times[nearest]-time)) > .02:
            raise ValueError('camera/RGB synchronization exceeds 20ms')
        # Read rows individually: repeated nearest indices are legal.
        r = np.stack([src['cam-pose/rotations'][int(i)] for i in nearest])
        t = np.stack([src['cam-pose/translations'][int(i)] for i in nearest])
        c2w = np.tile(np.eye(4),(len(frames),1,1))
        c2w[:,:3,:3], c2w[:,:3,3] = r @ rotation,t
        w2c = np.linalg.inv(c2w).astype(np.float32)
        depth = np.stack([cv2.remap(src['depth/frames'][int(i)],*dep_map,cv2.INTER_NEAREST) for i in frames]).astype(np.float32)/1000
        valid = src['depth/valid'][frames] & (np.abs(src['depth/timestamps'][frames]-time) < .02)
        mask = (depth > 0) & np.isfinite(depth) & valid[:,None,None]
        if not mask.reshape(len(frames),-1).any(-1).all():
            raise ValueError('sample contains frames without valid depth')
    Kall = np.tile(new_K.astype(np.float32),(len(frames),1,1))
    target = args.output/'stera_scene.npz'
    np.savez_compressed(target,rgb=np.stack(rgb),timestamp=time,frame_present=np.ones(len(frames),dtype=bool),
        target__scene_depth=depth,valid__scene_depth=mask,
        target__scene_K=Kall,valid__scene_K=np.ones_like(Kall,dtype=bool),
        target__scene_pose=w2c,valid__scene_pose=np.ones_like(w2c,dtype=bool))
    row = {'dataset':'STERA-10M','source_episode_id':args.session.name,'split':args.split,
           'path':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
           'adapter':'stera_scene_v1','source_frames':frames.tolist(),
           'scope':'real scene labels; no hand/contact labels',
           'source':str(args.session),'image_undistorted':True}
    (args.output/'index.jsonl').write_text(json.dumps(row)+'\n')
    (args.output/'preparation.json').write_text(json.dumps({'manifest':row,'K':new_K.tolist(),
        'optical_to_link':rotation.tolist(),'depth_valid_pixels':mask.sum(axis=(1,2)).tolist(),
        'max_pose_time_error_s':float(np.max(np.abs(pose_times[nearest]-time)))},indent=2))

if __name__ == '__main__':
    main()
