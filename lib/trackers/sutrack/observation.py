"""Runtime observation hooks; official tracker source and native decoding unchanged."""
from functools import partial as _qir_partial
from lib.trackers.checkpoint_io import load_checkpoint as _qir_load_checkpoint
_qir_torch_load = _qir_partial(_qir_load_checkpoint, kind='sutrack')
from .features import *
import sys,importlib,types
import torch,cv2

def tensorhash(v):
 h=hashlib.sha256()
 def add(x):
  if torch.is_tensor(x):h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str((str(x.dtype),x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x):h.update(str(k).encode());add(x[k])
  elif isinstance(x,(list,tuple)):
   for z in x:add(z)
  elif hasattr(x,'__dict__'):add(vars(x))
  else:h.update(repr(x).encode())
 add(v);return h.hexdigest()

class Adapter:
 def __init__(self,domain,checkpoint,config_file):
  kind='sutrack'
  self.kind=kind;self.capture=True;self.full=True;self.last={};self.last_crop=None
  self.repo=Path(__file__).resolve().parent/'native'
  checkpoint=Path(checkpoint)
  torch.set_num_threads(2);cv2.setNumThreads(0);torch.manual_seed(20261027);np.random.seed(20261027);random.seed(20261027)
  torch.backends.cudnn.benchmark=False
  state=_qir_torch_load(checkpoint,map_location='cpu')['net']
  if kind=='sutrack':
   import clip
   from clip.model import build_model
   sub={k[len('text_encoder.clip.'):]:v for k,v in state.items() if k.startswith('text_encoder.clip.')}
   assert sub and 'text_projection' in sub
   clip.load=lambda name,device='cpu',**kw:(build_model(sub).to(device),None)
   enc=importlib.import_module('lib.trackers.sutrack.native.models.sutrack.encoder');enc.is_main_process=lambda:False
  cfgmod=importlib.import_module('lib.trackers.sutrack.native.config.'+kind+'.config');cfgmod.update_config_from_file(str(config_file));cfg=cfgmod.cfg
  params=types.SimpleNamespace(cfg=cfg,checkpoint=str(checkpoint),debug=0,save_all_boxes=False,template_factor=cfg.TEST.TEMPLATE_FACTOR,template_size=cfg.TEST.TEMPLATE_SIZE,search_factor=cfg.TEST.SEARCH_FACTOR,search_size=cfg.TEST.SEARCH_SIZE)
  mod=importlib.import_module('lib.trackers.sutrack.native.test.tracker.'+kind)
  self.t=mod.get_tracker_class()(params,domain)
  result=self.t.network.load_state_dict(state,strict=True);assert not result.missing_keys and not result.unexpected_keys
  self.strict=dict(strict=True,keys=len(state),missing=[],unexpected=[],checkpoint=str(checkpoint),sha256=sha(checkpoint))
  assert self.strict['sha256']=='2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4'
  del state
  self.t.network.eval();self.t.network.requires_grad_(False)
  self.sample_module=mod;self.original_sample=mod.sample_target
  mod.sample_target=self.sample
  if kind=='ostrack':
   self.original_forward=self.t.network.forward
   def observed_forward(*args,**kwargs):
    out=self.original_forward(*args,**kwargs);self.observe_ost(self.t.network,args,out);return out
   self.t.network.forward=observed_forward
  else:
   self.hook=self.t.network.encoder.register_forward_hook(self.observe_su_enc)
   self.hook2=self.t.network.decoder.register_forward_hook(self.observe_su_head)
  self.static_buffer_hash=tensorhash(dict(self.t.network.named_buffers()))
 def sample(self,im,bbox,factor,output_sz=None,*args,**kw):
  result=self.original_sample(im,bbox,factor,output_sz,*args,**kw)
  if self.capture and output_sz==self.t.params.search_size:
   x,y,w,h=map(float,bbox);side=math.ceil(math.sqrt(w*h)*factor);x1=round(x+w/2-side/2);y1=round(y+h/2-side/2);H,W=im.shape[:2]
   pad=[max(0,-x1),max(0,-y1),max(x1+side-W+1,0),max(y1+side-H+1,0)]
   validmap=np.zeros((side,side),np.float32);l,t,r,b=pad;validmap[t:side-b if b else None,l:side-r if r else None]=1
   vf=cv2.resize(validmap,(24,24),interpolation=cv2.INTER_AREA)
   self.last_crop=dict(seed_bbox=list(map(float,bbox)),actual_xywh=[x1,y1,side,side],resize_factor=float(result[1]),padding_ltrb=pad,image_hw=[H,W],model_mapping_origin=[x+w/2-side/2,y+h/2-side/2],actual_to_model_origin_delta=[x1-(x+w/2-side/2),y1-(y+h/2-side/2)])
   self.last['search_valid_fraction']=vf;self.last['crop']=self.last_crop
  return result
 def observe_ost(self,m,args,out):
  if not self.capture:return
  z=out['backbone_feat'];z=z[-1] if isinstance(z,list) else z
  self.feat=z.detach();self.removed=np.zeros(576,np.bool_)
  for q in out.get('removed_indexes_s',[]):
   if q is not None:self.removed[q.detach().long().cpu().numpy().ravel()]=True
  self.head={k:out[k].detach() for k in ['score_map','size_map','offset_map']}
 def observe_su_enc(self,m,args,out):
  if self.capture:self.feat=out[0].detach();self.removed=np.zeros(576,np.bool_)
 def observe_su_head(self,m,args,out):
  if self.capture:self.head=dict(score_map=out[0].detach(),size_map=out[2].detach(),offset_map=out[3].detach())
 def initialize(self,image,bbox):
  self.capture=False;self.t.initialize(image,dict(init_bbox=list(map(float,bbox))));self.capture=True;self.template_frames=[None]* (self.t.num_template if self.kind=='sutrack' else 1)
 def snapshot(self):
  skip={'network','preprocessor','output_window'}
  return dict(fields=copy.deepcopy({k:v for k,v in vars(self.t).items() if k not in skip}),rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()),template_frames=list(self.template_frames))
 def restore(self,s):
  for k in list(vars(self.t)):
   if k not in {'network','preprocessor','output_window'}:delattr(self.t,k)
  for k,v in copy.deepcopy(s['fields']).items():setattr(self.t,k,v)
  random.setstate(s['rng'][0]);np.random.set_state(s['rng'][1]);torch.set_rng_state(s['rng'][2]);torch.cuda.set_rng_state_all(s['rng'][3]);self.template_frames=list(s['template_frames'])
 def digest(self,s):return tensorhash(s)
 def step(self,image,index,full=True):
  self.full=full;self.last={};old_templates=[v.data_ptr() for v in self.t.template_list] if self.kind=='sutrack' else []
  with torch.no_grad():out=self.t.track(image)
  bbox=list(map(float,out['target_bbox']));assert valid(bbox)
  if not self.capture:return out,None
  f=self.feat[0];ss=576
  if self.kind=='ostrack':search=f[-ss:];templates=f[:-ss]
  else:
   start=1 if self.t.network.class_token else 0;search=f[start:start+ss];templates=f[start+ss:start+ss+144*self.t.num_template]
  assert search.shape==(576,768) and templates.shape[-1]==768
  score=self.head['score_map'][0,0].float();response=score*self.t.output_window[0,0] if self.t.output_window.ndim==4 else score*self.t.output_window.squeeze()
  a=score.detach().cpu().numpy();q=a.ravel().clip(1e-12);q=q/q.sum();entropy=float(-(q*np.log(q)).sum()/np.log(576))
  origin=np.asarray(self.last_crop['model_mapping_origin']);side=self.last_crop['actual_xywh'][2]
  yy,xx=np.mgrid[:24,:24];cx=(xx+.5)*side/24+origin[0];cy=(yy+.5)*side/24+origin[1]
  roi=(cx>=bbox[0])&(cx<bbox[0]+bbox[2])&(cy>=bbox[1])&(cy<bbox[1]+bbox[3])&(~self.removed.reshape(24,24))&(self.last['search_valid_fraction']>.5)
  mask=torch.as_tensor(roi.ravel(),device=search.device);pool=search[mask].mean(0) if mask.any() else torch.zeros(768,device=search.device)
  obs=dict(frame_index=index,bbox_xywh=bbox,score_peak=float(a.max()),response_peak=float(response.max()),spatial_mass_entropy=entropy,roi_valid=bool(mask.any()),native_frame_id=int(self.t.frame_id),crop=self.last_crop,template_source_indices=list(self.template_frames))
  arrays=dict(score_map=a.astype(np.float16),response_map=response.detach().float().cpu().numpy().astype(np.float16),size_map=self.head['size_map'][0].float().cpu().numpy().astype(np.float16),offset_map=self.head['offset_map'][0].float().cpu().numpy().astype(np.float16),search_valid_fraction=self.last['search_valid_fraction'].astype(np.float16),search_ce_removed=self.removed,roi_mask=roi,roi_pool=pool.float().cpu().numpy().astype(np.float16),template_pool=templates.mean(0).float().cpu().numpy().astype(np.float16))
  if full:
   for name,feat in [('search',search),('template',templates)]:
    original=feat.float().cpu().numpy();qi,sc=quantize(original);arrays[name+'_tokens_qint8']=qi;arrays[name+'_tokens_scale']=sc
   rec=arrays['search_tokens_qint8'].astype(np.float32)*arrays['search_tokens_scale'];org=search.float().cpu().numpy();den=np.linalg.norm(org)*np.linalg.norm(rec)
   obs['quantization_search_cosine']=float((org*rec).sum()/den) if den>0 else 1.
  if self.kind=='sutrack':
   new=[v.data_ptr() for v in self.t.template_list]
   if new!=old_templates:self.template_frames=[self.template_frames[0]]+self.template_frames[2:]+[index]
  return obs,arrays
 def close(self):
  self.sample_module.sample_target=self.original_sample
  if self.kind=='ostrack':self.t.network.forward=self.original_forward
  else:self.hook.remove()
  if hasattr(self,'hook2'):self.hook2.remove()

def image(p):
 im=cv2.imread(str(p));assert im is not None,str(p);return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
