#!/usr/bin/env python3
"""torchrun entrypoint for bf16 ZeRO-2, with per-rank RNG and portable export.

recipe.batch_size is clips per rank per optimizer step. Microbatch=1 allows
different supervision keys; 8 accumulated clips x 8 GPUs gives effective64.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from egohandmetric_prompt.builders import build_runtime_marker_model
from egohandmetric_prompt.checkpoints import (load_marker_model_weights,
    MARKER_CHECKPOINT_SCHEMA_VERSION,MARKER_CHECKPOINT_NAMESPACE,ROOT_GEOMETRY_RUNTIME_CONTRACT)
from egohandmetric_prompt.configs import load_project_config,marker_architecture_descriptor,marker_architecture_hash
from egohandmetric_prompt.multirate_contracts import build_multirate_frame_map
from egohandmetric_prompt.training.data import PreparedClips
from egohandmetric_prompt.training.objectives import objective
from egohandmetric_prompt.training.scene import gather_scene_targets
from scripts.pwm.train_prepared import native_training_outputs,learning_rate


def zero_config(recipe,world_size):
    return {'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':recipe['batch_size'],
        'train_batch_size':recipe['batch_size']*world_size,'gradient_clipping':1.,
        'bf16':{'enabled':False},'torch_autocast':{'enabled':True,'dtype':'bfloat16'},
        'zero_optimization':{'stage':2,'overlap_comm':False,
            'contiguous_gradients':True,'reduce_bucket_size':50000000,
            'allgather_bucket_size':50000000,'ignore_unused_parameters':True},
        'steps_per_print':1000,'wall_clock_breakdown':False}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def export_model(model,config,path,step):
    payload={'checkpoint_schema_version':MARKER_CHECKPOINT_SCHEMA_VERSION,
        'checkpoint_namespace':MARKER_CHECKPOINT_NAMESPACE,
        'temporal_architecture':marker_architecture_descriptor(config.marker_model),
        'temporal_architecture_hash':marker_architecture_hash(config.marker_model),
        'root_geometry_contract':ROOT_GEOMETRY_RUNTIME_CONTRACT,
        'model':{k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.')},
        'step':step,'training_backend':'deepspeed_zero2',
        'resume_contract':'inference export only; resume from the sharded DeepSpeed directory'}
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','recipe','index','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--initialize',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if args.initialize and args.resume:
        parser.error('initialization and resume are mutually exclusive')
    import deepspeed
    from egohandmetric_prompt.training.zero_compat import install_scalar_broadcast_compat
    install_scalar_broadcast_compat()
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=int(os.environ.get('PWM_DISTRIBUTED_TIMEOUT_SECONDS','600'))))
    rank,world=dist.get_rank(),dist.get_world_size()
    recipe=json.loads(args.recipe.read_text())
    if not recipe.get('assumptions_acknowledged') or recipe['precision']!='bfloat16':
        raise ValueError('ZeRO-2 runner requires an explicit bf16 reconstruction recipe')
    if min(recipe[k] for k in ('steps','batch_size','save_every')) < 1:
        raise ValueError('positive steps, accumulation and checkpoint interval required')
    probabilities=recipe['stride_probabilities']
    if len(probabilities)!=5 or min(probabilities)<0 or abs(sum(probabilities)-1)>1e-6:
        raise ValueError('invalid stride distribution')
    random.seed(recipe['seed']+rank)
    np.random.seed(recipe['seed']+rank)
    torch.manual_seed(recipe['seed'])  # identical random-init model on every rank
    config=load_project_config(args.config)
    dataset=PreparedClips(args.index,'train')
    index_hash=hashlib.sha256(args.index.read_bytes()).hexdigest()
    model=build_runtime_marker_model(config)
    if args.initialize:
        load_marker_model_weights(model,args.initialize,model_config=config.marker_model,align_model_floating_dtype=True)
    model=model.float().train()
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError('backbone must remain frozen')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=recipe['lr'],
        betas=(.9,.999),eps=1e-8,weight_decay=recipe['weight_decay'])
    settings=zero_config(recipe,world)
    print(json.dumps({'rank':rank,'phase':'engine_initializing'}),flush=True)
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config=settings,dist_init_required=False)
    print(json.dumps({'rank':rank,'phase':'engine_ready'}),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=args.output/'sharded'
    start=0
    if args.resume:
        loaded,state=engine.load_checkpoint(str(checkpoint_dir))
        if not loaded:
            raise FileNotFoundError('no complete DeepSpeed checkpoint')
        if state['recipe']!=recipe or state['index_hash']!=index_hash or state['config']!=asdict(config) or state['world_size']!=world or state.get('engine_config')!=settings:
            raise ValueError('resume recipe/index/config/world size mismatch')
        restore_rng(state['rng_by_rank'][rank])
        start=state['step']
        if engine.global_steps!=start:
            raise ValueError('engine/trainer step mismatch')
    elif (checkpoint_dir/'latest').exists():
        raise FileExistsError('existing sharded run requires --resume')
    end=min(recipe['steps'],args.stop_after or recipe['steps'])
    if end<=start:
        raise ValueError('stop step must exceed resumed step')
    if rank==0:
        (args.output/'deepspeed_config.json').write_text(json.dumps(settings,indent=2))
    log=(args.output/f'rank{rank}.jsonl').open('a')
    for step in range(start,end):
        lr=learning_rate(step,recipe)
        for group in engine.optimizer.param_groups:
            group['lr']=lr
        reports=[]
        sources=[]
        for _ in range(recipe['batch_size']):
            sample=dataset[random.randrange(len(dataset))]
            sources.append(sample['identity']['dataset']+'/'+sample['identity']['source_episode_id'])
            images=sample['images'][None].to(engine.device)
            targets={k:(v[None].to(engine.device),m[None].to(engine.device)) for k,(v,m) in sample['targets'].items()}
            present=torch.ones(images.shape[:2],dtype=torch.bool,device=engine.device)
            stride=random.choices(range(1,6),weights=probabilities)[0]
            mapping=build_multirate_frame_map(high_frame_present=present,high_temporal_pair_valid=present[:,:-1],
                global_stride=stride,global_anchor_phase=min(2,stride-1))
            # The engine owns autocast; geometry modules explicitly retain fp32.
            print(json.dumps({'rank':rank,'step':step+1,'phase':'forward','source':sources[-1]}),flush=True)
            output=native_training_outputs(engine(images,multirate_frame_map=mapping))
            total,terms=objective(output,gather_scene_targets(targets,mapping),recipe['loss_weights'],
                confidence_alpha=recipe['confidence_alpha'],heatmap_positive_weight=recipe.get('heatmap_positive_weight',1.))
            # DeepSpeed alone scales accumulated losses and clips/reduces gradients.
            print(json.dumps({'rank':rank,'step':step+1,'phase':'backward'}),flush=True)
            engine.backward(total)
            print(json.dumps({'rank':rank,'step':step+1,'phase':'optimizer_step'}),flush=True)
            engine.step()
            reports.append({k:float(v.detach()) for k,v in terms.items()})
        if engine.global_steps!=step+1:
            raise RuntimeError('unexpected DeepSpeed accumulation boundary')
        record={'rank':rank,'world_size':world,'step':step+1,'lr':lr,'sources':sources,'terms':reports}
        log.write(json.dumps(record)+'\n'); log.flush()
        print(json.dumps(record),flush=True)
        if (step+1)%recipe['save_every']==0 or step+1==end:
            states=[None]*world
            dist.all_gather_object(states,rng_state())
            state={'step':step+1,'recipe':recipe,'index_hash':index_hash,'config':asdict(config),'engine_config':settings,
                   'world_size':world,'rng_by_rank':states}
            engine.save_checkpoint(str(checkpoint_dir),tag=f'step-{step+1}',client_state=state)
            if rank==0:
                export_model(engine.module,config,args.output/'inference.pt',step+1)
            dist.barrier()
    log.close()
    if rank==0:
        (args.output/'status.json').write_text(json.dumps({'state':'completed' if end==recipe['steps'] else 'stopped_at_requested_step',
            'steps':end,'world_size':world,'effective_batch_size':recipe['batch_size']*world}))
    dist.destroy_process_group()

if __name__=='__main__':
    main()
