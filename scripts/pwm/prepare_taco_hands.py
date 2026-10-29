#!/usr/bin/env python3
"""TACO V1 ZIP adapter using the dataset's pinned MANO implementation.

Only explicitly named members are read. Visibility is not fabricated from
in-frame projection; no visibility supervision is produced by this adapter.
"""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import pickle
import sys
import zipfile
import cv2
import numpy as np
import torch
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from egohandmetric_prompt.training.labels import hand_geometry, localization_labels
from egohandmetric_prompt.training.geometry import point_surface_distance, contact_from_distance
from egohandmetric_prompt.data.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--taco-code',type=Path,required=True)
    parser.add_argument('--mano-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--triplet',default='(dust, roller, pan)')
    parser.add_argument('--sequence',default='20230927_032')
    parser.add_argument('--start',type=int,default=30)
    parser.add_argument('--frames',type=int,default=3)
    args = parser.parse_args()
    # Compatibility for the original dataset's chumpy-based MANO loader.
    for name, value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
        if name not in np.__dict__:
            setattr(np,name,value)
    if not hasattr(inspect,'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    sys.path.insert(0,str(args.taco_code/'dataset_utils'))
    from manopth.manopth.manolayer import ManoLayer
    args.output.mkdir(parents=True,exist_ok=True)
    prefix = f'{args.triplet}/{args.sequence}'
    frames = np.arange(args.start,args.start+args.frames)
    hashes = {}
    def read(archive,member):
        with zipfile.ZipFile(args.root/archive) as file:
            info=file.getinfo(member)
            if info.file_size > 256*1024*1024:
                raise ValueError('sample member exceeds explicit 256 MiB limit')
            data=file.read(member)
        hashes[archive+'::'+member]=hashlib.sha256(data).hexdigest()
        return data
    camera_prefix=f'Egocentric_Camera_Parameters/{prefix}'
    K=np.loadtxt(io.BytesIO(read('Egocentric_Camera_Parameters_2.zip',camera_prefix+'/egocentric_intrinsic.txt')))
    poses=np.load(io.BytesIO(read('Egocentric_Camera_Parameters_2.zip',camera_prefix+'/egocentric_frame_extrinsic.npy')),allow_pickle=False)
    w2c=torch.from_numpy(poses[frames]).float()
    video=args.output/'source_color.mp4'
    video.write_bytes(read('Egocentric_RGB_Videos.zip',f'Egocentric_RGB_Videos/{prefix}/color.mp4'))
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(poses):
        raise ValueError('RGB/pose frame counts do not match')
    width,height=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    A=np.array([[256/width,0,(256/width-1)/2],[0,256/height,(256/height-1)/2],[0,0,1]])
    new_K=A@K
    rgb=[]
    for i in range(int(frames[-1])+1):
        ok,img=cap.read()
        if not ok: raise ValueError('video decode failed')
        if i >= args.start:
            rgb.append(cv2.cvtColor(cv2.resize(img,(256,256)),cv2.COLOR_BGR2RGB))
    cap.release()
    vertices,joints,faces=[],[],[]
    with torch.no_grad():
        for side in ('left','right'):
            # Trusted original dataset pickle contains torch tensors.
            hp=pickle.loads(read('Hand_Poses.zip',f'Hand_Poses/{prefix}/{side}_hand.pkl'))
            beta=pickle.loads(read('Hand_Poses.zip',f'Hand_Poses/{prefix}/{side}_hand_shape.pkl'))['hand_shape']
            keys=sorted(hp)
            if len(keys) != len(poses): raise ValueError('hand/RGB frame counts do not match')
            theta=torch.stack([torch.as_tensor(hp[keys[i]]['hand_pose']).reshape(48).cpu() for i in frames]).float()
            trans=torch.stack([torch.as_tensor(hp[keys[i]]['hand_trans']).reshape(3).cpu() for i in frames]).float()
            layer=ManoLayer(mano_root=str(args.mano_root),use_pca=False,ncomps=45,side=side,center_idx=0,flat_hand_mean=True)
            v,j,_=layer(theta,torch.as_tensor(beta).cpu().float().reshape(1,10).repeat(len(frames),1))
            v,j=v/1000+trans[:,None],j/1000+trans[:,None]
            vertices.append(torch.einsum('tij,tnj->tni',w2c[:,:3,:3],v)+w2c[:,:3,3,None].transpose(1,2))
            joints.append(torch.einsum('tij,tnj->tni',w2c[:,:3,:3],j)+w2c[:,:3,3,None].transpose(1,2))
            faces.append(layer.th_faces.long())
    v,j=torch.stack(vertices,1),torch.stack(joints,1)
    targets=hand_geometry(j,v,torch.isfinite(j).all(-1),torch.isfinite(v).all(-1))
    Kt=torch.tensor(new_K,dtype=torch.float32).repeat(len(frames),1,1)
    pix=torch.einsum('tij,tsnj->tsni',Kt,j)
    uv=pix[...,:2]/pix[...,2:].clamp_min(1e-6)
    preview=cv2.cvtColor(rgb[0],cv2.COLOR_RGB2BGR)
    for side,color in enumerate(((0,255,0),(0,128,255))):
        for x,y in uv[0,side].tolist():
            if np.isfinite(x+y) and 0 <= x < 256 and 0 <= y < 256:
                cv2.circle(preview,(round(x),round(y)),2,color,-1)
    cv2.imwrite(str(args.output/'label_projection.png'),preview)
    targets.update(localization_labels(uv,(j[...,2]>0)&torch.isfinite(j).all(-1),(256,256),(32,32)))
    targets['projection_K']=(Kt,torch.ones_like(Kt,dtype=torch.bool))
    object_surfaces=[[] for _ in frames]
    with zipfile.ZipFile(args.root/'Object_Poses.zip') as archive:
        names=[n for n in archive.namelist() if n.startswith(f'Object_Poses/{prefix}/') and n.endswith('.npy')]
    if len(names) != 2: raise ValueError('require exactly annotated tool and target')
    with zipfile.ZipFile(args.root/'Object_Models.zip') as archive:
        model_names=archive.namelist()
    for name in names:
        identifier=Path(name).stem.split('_')[-1]
        members=[m for m in model_names if Path(m).name==identifier+'_cm.obj']
        if len(members)!=1: raise ValueError('ambiguous object model '+identifier)
        mesh=trimesh.load(io.BytesIO(read('Object_Models.zip',members[0])),file_type='obj',force='mesh',process=False,skip_materials=True)
        objv=torch.tensor(np.asarray(mesh.vertices).copy(),dtype=torch.float32)*.01
        objf=torch.tensor(np.asarray(mesh.faces).copy(),dtype=torch.long)
        objpose=torch.from_numpy(np.load(io.BytesIO(read('Object_Poses.zip',name)),allow_pickle=False)[frames]).float()
        for fi in range(len(frames)):
            pose=w2c[fi]@objpose[fi]
            transformed=objv@pose[:3,:3].T+pose[:3,3]
            object_surfaces[fi].append(transformed[objf])
    ids=list(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)
    for kind,points in [('joint',j),('vertex',v[:,:,ids])]:
        distance=torch.empty(points.shape[:-1])
        for fi in range(len(frames)):
            for side in range(2):
                triangles=torch.cat(object_surfaces[fi]+[v[fi,1-side][faces[1-side]]])
                distance[fi,side]=point_surface_distance(points[fi,side].cuda(),triangles.cuda(),chunk_size=2048).cpu()
        contact,valid=contact_from_distance(distance,kind)
        targets[kind+'_distance']=(distance,valid)
        targets[kind+'_contact']=(contact.float(),valid)
    arrays={}
    for key,(value,mask) in targets.items():
        arrays['target__'+key]=value.numpy()
        arrays['valid__'+key]=mask.numpy()
    output=args.output/'taco_hands.npz'
    np.savez_compressed(output,rgb=np.stack(rgb),timestamp=frames/fps,frame_present=np.ones(len(frames),dtype=bool),**arrays)
    row={'dataset':'TACO','source_episode_id':prefix,'split':'train','path':output.name,
         'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'adapter':'taco_v1_hand_contact_v1','source_frames':frames.tolist()}
    (args.output/'index.jsonl').write_text(json.dumps(row)+'\n')
    report={'manifest':row,'source_sha256':hashes,'targets':list(targets),'K':new_K.tolist(),
        'paper_contact_labels':'union of tool, target and opposite hand triangles',
        'joint_depth_range_m':[float(j[...,2].min()),float(j[...,2].max())],
        'visibility':'not generated; unavailable annotation',
        'distortion':'TACO official ego rendering uses pinhole K; no separate D in camera release',
        'joint_contact_fraction':float(targets['joint_contact'][0].mean()),
        'vertex_contact_fraction':float(targets['vertex_contact'][0].mean())}
    (args.output/'preparation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
