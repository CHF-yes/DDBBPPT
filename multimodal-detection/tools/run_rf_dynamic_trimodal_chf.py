"""CHF image-dependent RGB/IR/depth gates; four aux epochs plus two head epochs."""
import argparse, contextlib, hashlib, json, math, random, shutil, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize
from rfdetr import RFDETRMedium
from rfdetr.config import TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.utilities.tensors import NestedTensor

SHAPE=(864,1536)
SPLIT_SHA='4165ce54e9b34be51ec3f87d3065977eee310da3ce700fa4b5b19a85202ba00a'

class Triples(Dataset):
    def __init__(self, data, coco, training=False):
        self.data=Path(data);self.coco=Path(coco);self.training=training
        info=json.loads(self.coco.read_text());self.images=info['images'];self.anns=defaultdict(list)
        assert {c['id'] for c in info['categories']}==set(range(12))
        for a in info['annotations']:self.anns[a['image_id']].append(a)
        for x in self.images:
            for mod in ['visible','infrared','depth']:
                if not (self.data/mod/x['file_name']).is_file():raise FileNotFoundError((mod,x['file_name']))
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        meta=self.images[i];name=meta['file_name']
        with Image.open(self.data/'visible'/name) as im:
            im=im.convert('RGB');w,h=im.size
            assert (w,h)==(meta['width'],meta['height']),('COCO dimensions',name,(w,h),meta)
            rgb=np.array(im.resize((SHAPE[1],SHAPE[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
        with Image.open(self.data/'infrared'/name) as im:
            ir=np.array(im.convert('L').resize((SHAPE[1],SHAPE[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
        with Image.open(self.data/'depth'/name) as im:dep=np.array(im)
        if dep.ndim==3:dep=dep[...,0]
        # Some source JPEG depth images are an 8-bit encoding, not millimetres.
        encoded8=dep.dtype==np.uint8
        valid=(dep>0) if encoded8 else ((dep>=300)&(dep<=20000))
        dep=dep.astype(np.float32)/(255 if encoded8 else 20000)
        dep=np.where(valid,np.clip(dep,0,1),0).astype(np.float32)
        dep=np.array(Image.fromarray(dep).resize((SHAPE[1],SHAPE[0]),Image.Resampling.NEAREST))
        mask=np.array(Image.fromarray(valid.astype(np.uint8)).resize((SHAPE[1],SHAPE[0]),Image.Resampling.NEAREST),dtype=np.float32)
        boxes=[];labels=[]
        for a in self.anns[meta['id']]:
            x,y,bw,bh=a['bbox'];boxes.append([(x+bw/2)/w,(y+bh/2)/h,bw/w,bh/h]);labels.append(a['category_id'])
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        if self.training and random.random()<.5:
            rgb=rgb[:,::-1].copy();ir=ir[:,::-1].copy();dep=dep[:,::-1].copy();mask=mask[:,::-1].copy()
            boxes[:,0]=1-boxes[:,0]
        rgb=normalize(torch.from_numpy(rgb.copy()).permute(2,0,1),[.485,.456,.406],[.229,.224,.225])
        ir=torch.from_numpy(ir.copy())[None];ir=(ir-.5)/.25
        dep=torch.from_numpy(np.stack([dep,mask]).copy())
        target={'labels':torch.tensor(labels,dtype=torch.long),'boxes':boxes,'image_id':torch.tensor(meta['id']),'orig_size':torch.tensor([h,w])}
        return rgb,ir,dep,target

def collate(rows):
    r,i,d,t=zip(*rows);return torch.stack(r),torch.stack(i),torch.stack(d),list(t)

class AuxEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__();layers=[]
        for oc in [16,32,64,96]:
            layers.extend([nn.Conv2d(channels,oc,3,stride=2,padding=1),nn.GroupNorm(4,oc),nn.SiLU()]);channels=oc
        self.body=nn.Sequential(*layers)
    def forward(self,x):return self.body(x)

class ResidualFusion(nn.Module):
    def __init__(self,channels):
        super().__init__();self.ir=AuxEncoder(1);self.depth=AuxEncoder(2)
        self.ir_proj=nn.ModuleList([nn.Conv2d(96,c,1) for c in channels])
        self.depth_proj=nn.ModuleList([nn.Conv2d(96,c,1) for c in channels])
        self.gates=nn.Parameter(torch.zeros(len(channels),2))
        self.dynamic=nn.ModuleList([nn.Sequential(nn.Linear(c*3,32),nn.SiLU(),nn.Linear(32,2)) for c in channels])
        for gate in self.dynamic:
            nn.init.zeros_(gate[-1].weight);nn.init.zeros_(gate[-1].bias)
    def forward(self,features,ir,depth,use_ir=True,use_depth=True):
        a=self.ir(ir) if use_ir else None;b=self.depth(depth) if use_depth else None
        gates=.2*self.gates.tanh();out=[]
        for k,f in enumerate(features):
            x=f.tensors
            ia=F.interpolate(self.ir_proj[k](a),size=x.shape[-2:],mode='bilinear',align_corners=False) if a is not None else torch.zeros_like(x)
            db=F.interpolate(self.depth_proj[k](b),size=x.shape[-2:],mode='bilinear',align_corners=False) if b is not None else torch.zeros_like(x)
            pooled=torch.cat([v.mean((-2,-1)) for v in (x,ia,db)],dim=1)
            gate=.2*torch.tanh(self.gates[k][None]+self.dynamic[k](pooled))
            if a is not None:x=x+gate[:,0,None,None,None]*ia
            if b is not None:x=x+gate[:,1,None,None,None]*db
            out.append(NestedTensor(x,f.mask,f.no_padding))
        return out

class TriModel(nn.Module):
    def __init__(self,detector,channels):
        super().__init__();self.net=detector.model.model;self.fusion=ResidualFusion(channels);self.current=None
        self.use_ir=True;self.use_depth=True;self.handle=self.net.backbone.register_forward_hook(self.hook)
    def hook(self,module,inputs,outputs):
        if self.current is None:return outputs
        ir,depth=self.current;features,pos,cross=outputs
        return self.fusion(features,ir,depth,self.use_ir,self.use_depth),pos,cross
    def forward(self,rgb,ir,depth):
        assert tuple(rgb.shape[-2:])==SHAPE
        self.current=(ir,depth)
        try:return self.net(rgb)
        finally:self.current=None

@torch.no_grad()
def evaluate(model,post,loader,coco_path,limit=0):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval();model.net.eval();pred=[];ids=[]
    for step,(r,i,d,targets) in enumerate(loader):
        if limit and step>=limit:break
        r,i,d=r.cuda(),i.cuda(),d.cuda();outputs=model(r,i,d)
        sizes=torch.stack([t['orig_size'] for t in targets]).cuda();res=post(outputs,sizes)
        for q,t in zip(res,targets):
            image_id=int(t['image_id']);ids.append(image_id);h,w=map(int,t['orig_size'])
            count=0
            for j in q['scores'].argsort(descending=True).tolist():
                c=int(q['labels'][j]);s=float(q['scores'][j]);box=q['boxes'][j].float().cpu().numpy()
                box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
                if not 0<=c<12 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
                pred.append({'image_id':image_id,'category_id':c,'bbox':[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],'score':s})
                count+=1
                if count==100:break
    coco=COCO(str(coco_path));ev=COCOeval(coco,coco.loadRes(pred),'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    return {'map50_95':float(ev.stats[0]),'map50':float(ev.stats[1])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--data',default='/root/autodl-tmp/data/train_extracted');p.add_argument('--coco',default='/root/autodl-tmp/data/chf_arch_rgb_coco_s42');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    coco=Path(a.coco);manifest=json.loads((coco/'export_manifest.json').read_text());assert manifest['split_sha256']==SPLIT_SHA
    training=Triples(a.data,coco/'train/_annotations.coco.json',True);validation=Triples(a.data,coco/'valid/_annotations.coco.json')
    assert len(training)==1600 and len(validation)==400
    train=DataLoader(training,batch_size=1,shuffle=True,num_workers=4,collate_fn=collate,pin_memory=True)
    val=DataLoader(validation,batch_size=1,num_workers=4,collate_fn=collate,pin_memory=True)
    snapshot=out/'rgb_best.pth';shutil.copy2(a.weights,snapshot)
    detector=RFDETRMedium.from_checkpoint(snapshot,device='cuda');net=detector.model.model.cuda().eval()
    for param in net.parameters():param.requires_grad_(False)
    sample=next(iter(val));r,i,d=[x.cuda() for x in sample[:3]];features=[]
    handle=net.backbone.register_forward_hook(lambda module,inputs,outputs:features.extend(outputs[0]))
    with torch.no_grad():reference=net(r)
    handle.remove();channels=[f.tensors.shape[1] for f in features]
    model=TriModel(detector,channels).cuda().eval()
    with torch.no_grad():zero=model(r,i,d)
    errors={k:float((zero[k]-reference[k]).abs().max()) for k in ['pred_logits','pred_boxes']};assert max(errors.values())==0,errors
    print('CHF zero-gate exact RGB equivalence:',errors,'channels',channels,flush=True)
    cfg=TrainConfig(dataset_dir=str(coco),output_dir=str(out),epochs=4,batch_size=1,grad_accum_steps=8,num_workers=4,seed=42)
    criterion,post=build_criterion_from_config(detector.model_config,cfg);criterion.eval()
    params=list(model.fusion.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    record={'canvas':SHAPE,'split_sha256':SPLIT_SHA,'rgb_best':str(snapshot),'channels':channels,'zero_equivalence':errors,'batch':1,'accumulate':8,'precision':'fp32','seed':42,'augmentation':'three-modal synchronized horizontal flip p=.5; fixed resize; no crop/color jitter','frozen_rgb':True,'train_groups':1,'aux_epochs':4,'head_epochs':2,'gating':'image-dependent RGB/IR/depth pooled features, zero initialized' ,'max_predictions':100,'test_used_for_training':False,'depth':'uint16 mm clipped at20m + valid mask; uint8 depth treated as encoded depth/255'}
    (out/'manifest.json').write_text(json.dumps(record,indent=2))
    # Verify native matching/loss and a real auxiliary gradient, without updating the model.
    targets=[{k:v.cuda() for k,v in t.items()} for t in sample[3]]
    lossdict=criterion(model(r,i,d),targets);loss=sum(v*criterion.weight_dict[k] for k,v in lossdict.items() if k in criterion.weight_dict)
    assert torch.isfinite(loss);loss.backward();g=model.fusion.gates.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().max()>0
    print('CHF smoke backward loss',float(loss),'gate gradient',g.detach().cpu().tolist(),flush=True);opt.zero_grad(set_to_none=True)
    if a.smoke:
        # Gates learn first; after one update gradients must reach both encoders.
        loss=sum(v*criterion.weight_dict[k] for k,v in criterion(model(r,i,d),targets).items() if k in criterion.weight_dict)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        loss=sum(v*criterion.weight_dict[k] for k,v in criterion(model(r,i,d),targets).items() if k in criterion.weight_dict)
        loss.backward()
        for encoder in [model.fusion.ir,model.fusion.depth]:
            grads=[p.grad for p in encoder.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(v).all() for v in grads) and any(v.abs().max()>0 for v in grads)
        print('CHF smoke both auxiliary encoders receive gradients',flush=True)
        (out/'smoke.json').write_text(json.dumps({'loss':float(loss),'equivalence':errors,'gate_grad':g.detach().cpu().tolist()}));return
    start=time.time();history=[]
    def status(state,**kw):(out/'status.json').write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2))
    status('baseline_validation')
    baseline=evaluate(model,post,val,validation.coco);history.append(dict(stage='baseline',epoch=0,**baseline));best=baseline['map50_95']
    (out/'metrics.json').write_text(json.dumps(history,indent=2))
    # Preserve an integrated zero-gate checkpoint even if learned fusion never wins.
    base=torch.load(snapshot,map_location='cpu',weights_only=False)
    def save(name,epoch,metrics):
        payload=dict(base);payload['model']={k:v.detach().cpu() for k,v in net.state_dict().items()};payload['chf_fusion_type']='dynamic_v1';payload['chf_aux']={k:v.detach().cpu() for k,v in model.fusion.state_dict().items()};payload['chf_channels']=channels;payload['chf_manifest']=record;payload['chf_epoch']=epoch;payload['chf_metrics']=metrics
        torch.save(payload,out/name)
    save('best.pth',0,baseline)
    for epoch in range(1,7):
        if epoch==5:
            if False:  # This experiment explicitly evaluates head adaptation regardless of aux gain.
                print('CHF no clear validation gain; skip conditional head fine-tuning',flush=True);break
            aux_best=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
            net.load_state_dict(aux_best['model']);model.fusion.load_state_dict(aux_best['chf_aux'])
            for name,param in net.named_parameters():
                if name.startswith(('class_embed.','bbox_embed.','transformer.enc_out_class_embed.','transformer.enc_out_bbox_embed.')):param.requires_grad_(True)
            params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW([{'params':model.fusion.parameters(),'lr':5e-5},{'params':[p for p in net.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=1e-4)
        model.fusion.train();net.eval();opt.zero_grad(set_to_none=True);total=0.;status('training',epoch=epoch,best=best)
        for step,(r,i,d,targets) in enumerate(train):
            r,i,d=r.cuda(),i.cuda(),d.cuda();targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
            losses=criterion(model(r,i,d),targets);loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at {epoch}/{step}')
            total+=float(loss.detach());(loss/8).backward()
            if (step+1)%8==0:
                nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
            if step%100==0:print(f'CHF epoch={epoch} step={step}/{len(train)} loss={float(loss):.5f} gates={(.2*model.fusion.gates.tanh()).detach().cpu().tolist()}',flush=True)
        status('validation',epoch=epoch,best=best);metric=evaluate(model,post,val,validation.coco);row=dict(epoch=epoch,stage='aux' if epoch<=4 else 'heads',loss=total/len(train),elapsed=time.time()-start,**metric);history.append(row)
        save('last.pth',epoch,metric)
        if metric['map50_95']>best:best=metric['map50_95'];save('best.pth',epoch,metric)
        (out/'metrics.json').write_text(json.dumps(history,indent=2));print('CHF epoch result',row,'best',best,flush=True)
    saved=torch.load(out/'best.pth',map_location='cpu',weights_only=False);net.load_state_dict(saved['model']);model.fusion.load_state_dict(saved['chf_aux'])
    ablations={}
    for tag,use_ir,use_depth in [('both',True,True),('rgb_only',False,False),('no_ir',False,True),('no_depth',True,False)]:
        model.use_ir=use_ir;model.use_depth=use_depth;ablations[tag]=evaluate(model,post,val,validation.coco)
    (out/'ablations.json').write_text(json.dumps(ablations,indent=2));status('complete',best=best,best_epoch=saved['chf_epoch'],baseline=baseline,ablations=ablations)
    print('CHF fusion complete',best,ablations,flush=True)
if __name__=='__main__':main()
