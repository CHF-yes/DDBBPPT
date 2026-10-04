"""Controlled CHF_v3 local gates / last-two RGB layer experiment."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from run_rf_dynamic_trimodal_chf import (Triples,collate,TriModel,ResidualFusion,NestedTensor,SHAPE,SPLIT_SHA,RFDETRMedium,TrainConfig,build_criterion_from_config,evaluate)
SOURCE=Path('/root/autodl-tmp/weights/CHF_v3/CHF_v3.pth')
SHA='936755d6f850ce7b19b12deb27b3af52df804ef4559f93cdabe1719b8ca7e977'
class LocalFusion(ResidualFusion):
 def __init__(self,channels):
  super().__init__(channels)
  self.local=nn.ModuleList([nn.Sequential(nn.Conv2d(c*3,32,1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.SiLU(),nn.Conv2d(32,2,1)) for c in channels])
  for m in self.local:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
 def forward(self,features,ir,depth,use_ir=True,use_depth=True):
  a=self.ir(ir) if use_ir else None;b=self.depth(depth) if use_depth else None;out=[]
  for k,f in enumerate(features):
   x=f.tensors
   ia=F.interpolate(self.ir_proj[k](a),size=x.shape[-2:],mode='bilinear',align_corners=False) if a is not None else torch.zeros_like(x)
   db=F.interpolate(self.depth_proj[k](b),size=x.shape[-2:],mode='bilinear',align_corners=False) if b is not None else torch.zeros_like(x)
   pooled=torch.cat([v.mean((-2,-1)) for v in (x,ia,db)],1)
   global_gate=.2*torch.tanh(self.gates[k][None]+self.dynamic[k](pooled))
   local_gate=.1*torch.tanh(self.local[k](torch.cat([x,ia,db],1)))
   if a is not None:x=x+global_gate[:,0,None,None,None]*ia+local_gate[:,0:1]*ia
   if b is not None:x=x+global_gate[:,1,None,None,None]*db+local_gate[:,1:2]*db
   out.append(NestedTensor(x,f.mask,f.no_padding))
  return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--arm',choices=['control','local','local_rgb'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',required=True);p.add_argument('--coco',type=Path,required=True);p.add_argument('--epochs',type=int,default=6);a=p.parse_args();global SOURCE;SOURCE=a.weights
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
 coco=a.coco;assert json.loads((coco/'export_manifest.json').read_text())['split_sha256']==SPLIT_SHA
 tr=Triples(a.data,coco/'train/_annotations.coco.json',True);va=Triples(a.data,coco/'valid/_annotations.coco.json');assert (len(tr),len(va))==(1600,400)
 train=DataLoader(tr,batch_size=1,shuffle=True,num_workers=4,collate_fn=collate,pin_memory=True);val=DataLoader(va,batch_size=1,num_workers=4,collate_fn=collate,pin_memory=True)
 base=torch.load(SOURCE,map_location='cpu',weights_only=False);detector=RFDETRMedium.from_checkpoint(SOURCE,device='cuda');net=detector.model.model.cuda().eval();net.load_state_dict(base['model'],strict=True)
 model=TriModel(detector,base['chf_channels']).cuda().eval();model.fusion.load_state_dict(base['chf_aux'],strict=True)
 r,i,d,t=next(iter(val));r,i,d=r.cuda(),i.cuda(),d.cuda()
 with torch.no_grad():ref=model(r,i,d)
 if a.arm!='control':
  new=LocalFusion(base['chf_channels']).cuda();missing,extra=new.load_state_dict(base['chf_aux'],strict=False)
  assert not extra and missing and all(k.startswith('local.') for k in missing)
  model.fusion=new
 with torch.no_grad():actual=model(r,i,d)
 errors={k:float((actual[k]-ref[k]).abs().max()) for k in ('pred_logits','pred_boxes')};assert max(errors.values())==0,errors
 head_prefix=('class_embed.','bbox_embed.','transformer.enc_out_class_embed.','transformer.enc_out_bbox_embed.')
 rgb_prefix=('backbone.0.encoder.encoder.encoder.layer.10.','backbone.0.encoder.encoder.encoder.layer.11.')
 heads=[];rgb=[];rgb_names=[]
 for name,param in net.named_parameters():
  head=name.startswith(head_prefix);last=a.arm=='local_rgb' and name.startswith(rgb_prefix);param.requires_grad_(head or last)
  if head:heads.append(param)
  if last:rgb.append(param);rgb_names.append(name)
 assert heads and (a.arm!='local_rgb' or rgb)
 groups=[{'params':model.fusion.parameters(),'lr':2e-5},{'params':heads,'lr':5e-6}]
 if rgb:groups.append({'params':rgb,'lr':1e-6})
 opt=torch.optim.AdamW(groups,weight_decay=1e-4);params=[v for v in model.parameters() if v.requires_grad]
 cfg=TrainConfig(dataset_dir=str(coco),output_dir=str(out),epochs=a.epochs,batch_size=1,grad_accum_steps=8,num_workers=4,seed=42);criterion,post=build_criterion_from_config(detector.model_config,cfg);criterion.eval()
 manifest=dict(source=str(SOURCE),source_sha256=SHA,arm=a.arm,canvas=SHAPE,split_sha256=SPLIT_SHA,train=1600,validation=400,epochs=a.epochs,seed=42,precision='FP32',batch=1,accumulate=8,aux_lr=2e-5,heads_lr=5e-6,rgb_lr=1e-6 if rgb else None,unfrozen_rgb=rgb_names,equivalence=errors,augmentation='fixed resize; synchronized flip .5',max_predictions=100,test_used=False,selection='held-out pycoco AP95; compare baseline and control; no test tuning')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time();history=[]
 def status(state,**kw):(out/'status.json').write_text(json.dumps(dict(state=state,time=time.time(),elapsed=time.time()-start,**kw),indent=2))
 # Reset training RNG after different module constructors for matched ordering/flip.
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 if a.smoke:
  opt.zero_grad(set_to_none=True)
  for step,(r,i,d,ts) in enumerate(train):
   ts=[{k:v.cuda() for k,v in t.items()} for t in ts];ld=criterion(model(r.cuda(),i.cuda(),d.cuda()),ts);loss=sum(v*criterion.weight_dict[k] for k,v in ld.items() if k in criterion.weight_dict);assert torch.isfinite(loss);(loss/8).backward()
   if step==7:break
  for name,module in [('ir',model.fusion.ir),('depth',model.fusion.depth)]+([('local',model.fusion.local)] if a.arm!='control' else []):
   gs=[v.grad for v in module.parameters() if v.grad is not None];assert gs and all(torch.isfinite(g).all() for g in gs) and any(g.abs().max()>0 for g in gs),name
  if rgb:assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().max()>0 for v in rgb)
  nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);opt.step();status('smoke_passed',equivalence=errors,max_memory=torch.cuda.max_memory_allocated(),loss=float(loss));print('SMOKE PASSED',a.arm,errors,flush=True);return
 status('baseline_validation');baseline=evaluate(model,post,val,va.coco);best=baseline['map50_95'];history.append(dict(epoch=0,**baseline))
 def save(name,epoch,metric):
  payload=dict(base);payload.update(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},chf_aux={k:v.detach().cpu().clone() for k,v in model.fusion.state_dict().items()},chf_fusion_type='dynamic_v1' if a.arm=='control' else 'local_v1',chf_manifest=manifest,chf_epoch=epoch,chf_metrics=metric);torch.save(payload,out/name)
 save('best.pth',0,baseline)
 for epoch in range(1,a.epochs+1):
  net.eval();model.fusion.train();total=0;opt.zero_grad(set_to_none=True)
  for step,(r,i,d,ts) in enumerate(train):
   assert tuple(r.shape[-2:])==SHAPE;ts=[{k:v.cuda() for k,v in t.items()} for t in ts];ld=criterion(model(r.cuda(),i.cuda(),d.cuda()),ts);loss=sum(v*criterion.weight_dict[k] for k,v in ld.items() if k in criterion.weight_dict);assert torch.isfinite(loss),(epoch,step);total+=float(loss.detach());(loss/8).backward()
   if (step+1)%8==0:nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
   if step%100==0:status('training',epoch=epoch,batch=step+1,total_batches=1600,best=best,loss=float(loss));print('CHF UPGRADE',a.arm,epoch,step+1,float(loss),flush=True)
  status('validation',epoch=epoch);metric=evaluate(model,post,val,va.coco);row=dict(epoch=epoch,loss=total/1600,elapsed=time.time()-start,**metric);history.append(row);save('last.pth',epoch,metric)
  if metric['map50_95']>best:best=metric['map50_95'];save('best.pth',epoch,metric)
  (out/'metrics.json').write_text(json.dumps(history,indent=2));print('EPOCH',row,flush=True)
 saved=torch.load(out/'best.pth',map_location='cpu',weights_only=False);net.load_state_dict(saved['model']);model.fusion.load_state_dict(saved['chf_aux']);ablations={}
 for tag,ir,dep in [('both',True,True),('rgb_only',False,False),('no_ir',False,True),('no_depth',True,False)]:
  model.use_ir=ir;model.use_depth=dep;ablations[tag]=evaluate(model,post,val,va.coco)
 (out/'ablations.json').write_text(json.dumps(ablations,indent=2));status('complete',best=best,best_epoch=saved['chf_epoch'],baseline=baseline,ablations=ablations)
if __name__=='__main__':main()
