"""Legacy AR B256 GOT paired S9 fine-tuning and offline evaluation, NOT tracking."""
from pathlib import Path
import sys,argparse,json,random,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lib.models.qirtrack.checkpoint import load_legacy_diagnostic
from lib.models.qirtrack.core.trigger_e2e.losses import IoUDistributionLoss
from lib.train import legacy_pair_helpers as h

def run():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--run-name',required=True);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=20261029);p.add_argument('--q',type=float,default=.4);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 if a.steps<1 or a.batch_size<1 or not 0<a.q<1:p.error('invalid steps, batch-size or threshold')
 cfg=json.loads(Path(a.config).read_text()); root=Path(cfg['data_root']).resolve(); manifest=Path(a.manifest).resolve()
 groups={};seen=set()
 for line in manifest.read_text().splitlines():
  row=json.loads(line)
  if not row['label_valid']:continue
  if row['sample_id'] in seen:raise ValueError('duplicate pair identity')
  seen.add(row['sample_id'])
  if not all(0<=row[k]<=1 for k in ['raw_iou','candidate_iou']):raise ValueError('invalid IoU label')
  if abs(row['true_gain']-row['candidate_iou']+row['raw_iou'])>1e-8:raise ValueError('inconsistent gain label')
  if row['origin'] not in ['actual_base_local_trial','actual_mllm_global_base_trial']:raise ValueError('unsupported candidate origin')
  for k in ['raw_inputs','candidate_inputs']:
   name=row[k]['file']
   for old,new in sorted(cfg.get('path_map',{}).items(),key=lambda x:-len(x[0])):
    if name==old or name.startswith(old.rstrip('/')+'/'):name=new+name[len(old):];break
   path=Path(name);path=(root/path).resolve() if not path.is_absolute() else path.resolve()
   if root!=path and root not in path.parents:raise ValueError('input outside configured data root; configure path_map')
   if not path.is_file():raise FileNotFoundError(path)
   row[k]['file']=str(path)
  groups.setdefault(row['sequence'],[]).append(row)
 if not groups:raise ValueError('empty valid pair set')
 rows=[x for seq in groups.values() for x in seq]
 cache=h.Cache(torch)
 for ref in [rows[0]['raw_inputs'],rows[0]['candidate_inputs']]:cache.get(ref)
 model,checkpoint=load_legacy_diagnostic(cfg['checkpoint'],'cpu')
 if a.validate_only:
  print(json.dumps(dict(valid_pairs=len(rows),sequences=len(groups),strict_checkpoint=True,first_pair_hash_and_shapes=True,all_payloads_verified=False)));return
 if not torch.cuda.is_available():raise RuntimeError('CUDA required by PreciseRoIPooling')
 if a.mode=='train' and a.batch_size>len(groups):raise ValueError('batch-size exceeds sequence count')
 out=Path(cfg['output_root'])/a.run_name;out.mkdir(parents=True,exist_ok=False)
 (out/'protocol.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=h.sha(manifest),checkpoint_sha256=h.sha(cfg['checkpoint']),diagnostic_only=True,sampling='original sequence/raw-IoU/gain/global stratified schedule',scope='paired fine-tuning or offline predictor evaluation; not end-to-end tracking'),indent=2))
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 model.cuda()
 if a.mode=='train':
  gs=[dict(sequence=s,pairs=v,local=[i for i,r in enumerate(v) if r['origin']=='actual_base_local_trial'],global_=[i for i,r in enumerate(v) if r['origin']=='actual_mllm_global_base_trial']) for s,v in groups.items()]
  h.UPDATES=a.steps;h.BATCH=a.batch_size;draws,counts=h.schedule(gs,a.seed)
  (out/'sampling.json').write_text(json.dumps(counts,indent=2));model.train()
  optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=1e-4);objective=IoUDistributionLoss()
  with (out/'steps.jsonl').open('x') as log:
   for step,draw in enumerate(draws,1):
    pairs=[gs[g]['pairs'][i] for g,i in draw];x=h.batch(cache,[r[k] for r in pairs for k in ['raw_inputs','candidate_inputs']],torch)
    y=torch.tensor([[r['raw_iou'],r['candidate_iou']] for r in pairs],device='cuda');optimizer.zero_grad(set_to_none=True)
    pred=model(x);q=pred['predicted_iou'].reshape(-1,2)
    absolute=objective(pred['iou_logits'].reshape(-1,2,50),y)['loss'];relative=torch.nn.functional.smooth_l1_loss(q[:,1]-q[:,0],y[:,1]-y[:,0]);loss=absolute+10*relative
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();assert all(v.grad is None for v in model.base_model.parameters())
    torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True);optimizer.step()
    log.write(json.dumps(dict(step=step,loss=float(loss.detach()),pair_ids=[r['sample_id'] for r in pairs]))+'\n');log.flush()
  checkpoint.update(model=model.state_dict(),optimizer=optimizer.state_dict(),parent_step=checkpoint['step'],step=checkpoint['step']+a.steps,new_updates=a.steps,release_role='user_finetuned_diagnostic',manifest_sha256=h.sha(manifest))
  torch.save(checkpoint,out/'checkpoint.pt')
 else:
  model.eval();raws={};candidate_errors=[];wins=ties=total=0
  with torch.no_grad(),(out/'pairs.jsonl').open('x') as f:
   for start in range(0,len(rows),a.batch_size):
    pairs=rows[start:start+a.batch_size];x=h.batch(cache,[r[k] for r in pairs for k in ['raw_inputs','candidate_inputs']],torch);q=model(x)['predicted_iou'].reshape(-1,2).cpu().tolist()
    for r,values in zip(pairs,q):
     key=(r['sequence'],r['frame']);entry=(r['raw_inputs']['sha256'],values[0],r['raw_iou'])
     if key in raws and (raws[key][0]!=entry[0] or abs(raws[key][2]-entry[2])>1e-8):raise ValueError('conflicting raw identity')
     raws[key]=entry;candidate_errors.append(abs(values[1]-r['candidate_iou']))
     gain=r['true_gain'];predgain=values[1]-values[0]
     if abs(gain)>1e-8:total+=1;wins+=int(predgain*gain>0);ties+=int(abs(predgain)<=1e-8)
     f.write(json.dumps(dict(sequence=r['sequence'],frame=r['frame'],sample_id=r['sample_id'],raw_q=values[0],candidate_q=values[1],raw_iou=r['raw_iou'],candidate_iou=r['candidate_iou']))+'\n')
  q=np.array([v[1] for v in raws.values()]);y=np.array([v[2] for v in raws.values()]);bad=y<.4;good=y>=.5;trig=q<a.q
  tp=int((trig&bad).sum());fp=int((trig&good).sum());fn=int((~trig&bad).sum())
  report=dict(raw_n=len(y),candidate_n=len(rows),raw_mae=float(abs(q-y).mean()),candidate_mae=float(np.mean(candidate_errors)),bad_n=int(bad.sum()),good_n=int(good.sum()),grey_excluded=int((~(bad|good)).sum()),q_threshold=a.q,recall=tp/int(bad.sum()) if bad.any() else None,fpr=fp/int(good.sum()) if good.any() else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,rank_correct=wins,rank_denominator=total,rank_prediction_ties=ties,rank= wins/total if total else None,scope='offline; does not report tracking AUC or FPS')
  (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
 print(out.resolve())
if __name__=='__main__':run()
