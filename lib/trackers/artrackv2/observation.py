"""Base256-seq-GOT observations with the original cached-input contract."""
from types import SimpleNamespace
from pathlib import Path
import hashlib,math,copy
import numpy as np
import torch
import torch.nn.functional as F
from . import state
from .capture import Capture,inputs_for
from .columns import Column
from .native.config.artrackv2_seq.config import cfg,update_config_from_file
from .native.test.tracker.artrackv2_seq import ARTrackV2Seq
from .native.train.data.processing_utils import sample_target,transform_image_to_crop
from lib.models.qirtrack.base_raw_model import make_sample
from lib.models.qirtrack.checkpoint import load_legacy_diagnostic
from lib.models.qirtrack.core.localisation.spatial_provenance import compute_search_crop_metadata,pooled_valid_fraction,xywh_image_to_search_normalized,validate_resize_factor
from lib.models.qirtrack.core.localisation.candidate_data import candidate_relative_geometry

def pool16(tokens):
    side=math.isqrt(tokens.shape[1]);assert tokens.shape[0]==1 and tokens.shape[2]==768
    return F.adaptive_avg_pool2d(tokens.transpose(1,2).reshape(1,768,side,side),(4,4)).flatten(2).transpose(1,2)[0]

class Adapter:
    def __init__(self,domain,checkpoint,config_file,predictor,policy,files,read_image):
        update_config_from_file(str(config_file));assert cfg.TEST.SEARCH_SIZE==256
        cfg.MODEL.PRETRAIN_PTH=str(checkpoint)
        h=hashlib.sha256()
        with open(checkpoint,'rb') as f:
            for b in iter(lambda:f.read(4194304),b''):h.update(b)
        assert h.hexdigest()=='6c2654caa850fce22ccde730e1040b72becb4364055cd044dea22322f677c604','This adapter is qualified for B256-seq-GOT only'
        params=SimpleNamespace(cfg=cfg,checkpoint=str(checkpoint),debug=0,save_all_boxes=False,template_factor=cfg.TEST.TEMPLATE_FACTOR,template_size=cfg.TEST.TEMPLATE_SIZE,search_factor=cfg.TEST.SEARCH_FACTOR,search_size=cfg.TEST.SEARCH_SIZE)
        self.t=ARTrackV2Seq(params,domain);self.t.network.requires_grad_(False)
        self.model,_=load_legacy_diagnostic(predictor,'cuda');self.policy=policy;self.files=files
        self.a=dict(np=np,torch=torch,setup=SimpleNamespace(pool16=pool16),base=SimpleNamespace(image=read_image),sample_target=sample_target,transform_image_to_crop=transform_image_to_crop,make_sample=make_sample,compute_search_crop_metadata=compute_search_crop_metadata,pooled_valid_fraction=pooled_valid_fraction,xywh_image_to_search_normalized=xywh_image_to_search_normalized,validate_resize_factor=validate_resize_factor,candidate_relative_geometry=candidate_relative_geometry)
        self.cap=Capture(self.t,self.a)
        self.native={k:Column(np) for k in ['frame','pred_bbox','tracker_score','coord_entropy_normalized','coord_logits','search_tokens','anchor_template_tokens']}
    def snapshot(self):
        s=state.capture(self.t);s.values['_embedding']=self.t.network.backbone.word_embeddings.weight.detach().clone();return s
    def restore(self,s):
        core=state.State({k:v for k,v in s.values.items() if k!='_embedding'},s.geometry,s.spec,s.parent_digest,s.rng);state.restore(self.t,core)
        with torch.no_grad():self.t.network.backbone.word_embeddings.weight.copy_(s.values['_embedding'])
    def initialize(self,image,box):self.t.initialize(image,dict(init_bbox=box))
    def score(self,box,capture,previous):
        try:
            x,_,_=inputs_for(self.t,self.cap,self.native,self.frame,self.image,previous,box,capture,self.before,self.files,self.policy,self.a)
            with torch.no_grad():return float(self.model({k:v[None].cuda() for k,v in x.items()})['predicted_iou'].item())
        finally:self.restore(self.post)
    def native_step(self,image,frame):
        self.image=image;self.frame=frame;self.before=self.snapshot();previous=np.asarray(self.t.state,dtype=np.float64)
        with torch.no_grad():box,capture=self.cap.step(image)
        self.post=self.snapshot();self.native['frame'].append(frame);self.native['pred_bbox'].append(box)
        for k in ['tracker_score','coord_entropy_normalized','coord_logits','search_tokens']:self.native[k].append(capture[k].numpy().astype(np.float16 if k in ['coord_logits','search_tokens'] else np.float32))
        self.native['anchor_template_tokens'].append(capture['template_tokens'].numpy().astype(np.float16))
        self.t.last_diagnostics=dict(coord_logits=capture['coord_logits'].numpy(),coord_argmax_index=capture['argmax'])
        return previous.tolist(),box.tolist(),self.score(box,capture,previous),float(capture['tracker_score'][0]),capture['entropy']
    def trial(self,seed,factor):
        try:
            self.restore(self.before);self.t.state=list(seed);self.t.params.search_factor=factor
            with torch.no_grad():box,capture=self.cap.step(self.image)
            return box.tolist(),self.score(box,capture,np.asarray(seed,dtype=np.float64))
        finally:
            self.restore(self.post);assert self.snapshot().digest==self.post.digest
    def assert_fallback(self):assert self.snapshot().digest==self.post.digest
    def close(self):self.t.network.backbone.forward=self.cap.bb;self.t.network.forward=self.cap.net
