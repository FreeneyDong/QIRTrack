"""Transactional mutable-state adapter for the frozen Base-256-seq tracker.

This does not generate boxes, implement a quality gate, or alter network weights.
Only an externally accepted candidate can enter the commit path. x_feat remains
the raw branch cache in this single-variable dz_feat ablation; the current network
does not consume its search_feature argument.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import random

import numpy as np
import torch

FIELDS=('state','frame_id','store_result','dz_feat','x_feat','update_',
        'z_dict1','z_dict2','box_mask_z','z_patch_arr')
GEOMETRY=('search_factor','search_size','template_factor','template_size')


def clone(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,list):return [clone(x) for x in value]
    if isinstance(value,tuple):return tuple(clone(x) for x in value)
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if hasattr(value,'tensors') and hasattr(value,'mask'):
        return type(value)(clone(value.tensors),clone(value.mask))
    return copy.deepcopy(value)


def serial(value):
    if torch.is_tensor(value):
        x=value.detach().cpu().contiguous()
        return dict(kind='tensor',shape=list(x.shape),dtype=str(x.dtype),sha=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):return dict(kind='array',shape=list(value.shape),dtype=str(value.dtype),sha=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,(list,tuple)):return [serial(x) for x in value]
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if hasattr(value,'tensors') and hasattr(value,'mask'):return dict(tensors=serial(value.tensors),mask=serial(value.mask))
    if isinstance(value,float):return dict(float_hex=value.hex())
    return value


@dataclass
class State:
    values: dict
    geometry: dict
    spec: dict
    parent_digest: str = None
    rng: dict = None

    @property
    def frame(self):return self.values['frame_id']

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(serial(dict(values=self.values,geometry=self.geometry,spec=self.spec,rng=self.rng)),sort_keys=True).encode()).hexdigest()


def capture(tracker,parent=None):
    assert not tracker.network.training, 'Frozen eval-mode tracker required'
    return State(values={k:clone(getattr(tracker,k)) for k in FIELDS},
        geometry={k:clone(getattr(tracker.params,k)) for k in GEOMETRY},
        spec={k:int(getattr(tracker,k)) for k in ['bins','range','prenum']},
        parent_digest=parent.digest if parent is not None else None,
        rng=dict(python=clone(random.getstate()),numpy=clone(np.random.get_state()),torch_cpu=torch.get_rng_state().clone(),
                 torch_cuda=torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None))


def restore(tracker,state):
    assert {k:int(getattr(tracker,k)) for k in state.spec}==state.spec, 'Architecture/state mismatch'
    values=clone(state.values);geometry=clone(state.geometry)
    for k,v in values.items():setattr(tracker,k,v)
    for k,v in geometry.items():setattr(tracker.params,k,v)
    if state.rng is not None:
        random.setstate(clone(state.rng['python']));np.random.set_state(clone(state.rng['numpy']))
        torch.set_rng_state(state.rng['torch_cpu'].clone())
        if state.rng['torch_cuda'] is not None:torch.cuda.set_rng_state(state.rng['torch_cuda'].clone())


def fork_into(native,trial,state):
    assert native is not trial and native.params is not trial.params, 'Trial aliases native object/params'
    restore(trial,state)


@contextmanager
def isolated_branch(tracker,before):
    """Sequential branch sharing frozen weights, always restoring native state."""
    resume=capture(tracker)
    try:
        restore(tracker,before)
        yield tracker
    finally:
        restore(tracker,resume)


def commit(tracker,before,raw,accepted_bbox=None,history='keep_raw',appearance='raw',trial=None,trusted=None,proof=None):
    """Commit at t without advancing time twice or rewinding the whole tracker.

    history: keep_raw or sync_current. appearance: raw, pre_frame, trial, trusted.
    trusted restores dz_feat only, not the old frame counter or bbox trajectory.
    """
    assert raw.frame==before.frame+1, 'Raw branch must advance exactly one frame'
    assert raw.spec==before.spec
    assert raw.parent_digest==before.digest, 'Raw branch lineage mismatch'
    assert history in ('keep_raw','sync_current') and appearance in ('raw','pre_frame','trial','trusted')
    target=State(clone(raw.values),clone(raw.geometry),clone(raw.spec),before.digest,clone(raw.rng))
    if accepted_bbox is not None:
        box=np.asarray(accepted_bbox,dtype=np.float64)
        assert box.shape==(4,) and np.isfinite(box).all() and (box[2:]>0).all(), 'Invalid accepted box'
        target.values['state']=box.tolist()
        if history=='sync_current':
            assert len(target.values['store_result'])==target.spec['prenum']
            target.values['store_result'][-1]=box.tolist()
        if appearance=='pre_frame':target.values['dz_feat']=clone(before.values['dz_feat'])
        elif appearance=='trial':
            assert trial is not None and trial.frame==raw.frame and trial.spec==raw.spec
            assert trial.parent_digest==before.digest, 'Trial must branch from pre-current-frame state'
            target.values['dz_feat']=clone(trial.values['dz_feat'])
        elif appearance=='trusted':
            assert trusted is not None and trusted.frame<=before.frame and trusted.spec==raw.spec
            assert proof and proof.get('released') is True and proof.get('invalidated') is False
            assert proof.get('source_frame')==trusted.frame and proof.get('request_ids'), 'Actual trusted-history admission proof required'
            target.values['dz_feat']=clone(trusted.values['dz_feat'])
    else:
        assert history=='keep_raw' and appearance=='raw', 'This accepted-event ablation does not change rejected frames'
    # All validation above is side-effect free; only now mutate the target tracker.
    restore(tracker,target)
    assert tracker.frame_id==raw.frame
    return capture(tracker,parent=before)
