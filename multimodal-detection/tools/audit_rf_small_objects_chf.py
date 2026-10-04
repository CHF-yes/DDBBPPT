"""Three-model small-object audit, fixed 400 images and global max100 predictions."""
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from rfdetr import RFDETRMedium
from rfdetr.config import TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config
from run_rf_trimodal_chf import Triples,collate,TriModel as StaticTriModel
from run_rf_dynamic_trimodal_chf import TriModel as DynamicTriModel

def diagnose(coco,pred):
    by_image=defaultdict(list)
    for p in pred:
        if p['score']>=.25:by_image[p['image_id']].append(p)
    summary={}
    for cid,cat in coco.cats.items():
        counts=dict(gt=0,gt_small=0,matched50=0,matched75=0,small_matched50=0,unmatched_predictions=0,localization_50_to_75=0)
        for iid in coco.imgs:
            gt=[g for g in coco.imgToAnns[iid] if g['category_id']==cid and not g.get('iscrowd',0)]
            ps=sorted([p for p in by_image[iid] if p['category_id']==cid],key=lambda p:-p['score'])
            counts['gt']+=len(gt); counts['gt_small']+=sum(g['area']<32**2 for g in gt)
            for threshold in [.5,.75]:
                used=set()
                for p in ps:
                    best=-1;score=0
                    x,y,w,h=p['bbox']
                    for j,g in enumerate(gt):
                        if j in used:continue
                        a,b,c,d=g['bbox'];inter=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
                        iou=inter/max(w*h+c*d-inter,1e-9)
                        if iou>score:best=j;score=iou
                    if score>=threshold:
                        used.add(best)
                        if threshold==.5 and gt[best]['area']<32**2:counts['small_matched50']+=1
                counts['matched50' if threshold==.5 else 'matched75']+=len(used)
                if threshold==.5:counts['unmatched_predictions']+=len(ps)-len(used)
        counts['localization_50_to_75']=counts['matched50']-counts['matched75']
        counts['missed_gt50']=counts['gt']-counts['matched50']
        counts['missed_small_gt50']=counts['gt_small']-counts['small_matched50']
        summary[cat['name']]=counts
    return summary
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
    result={'map50_95':float(ev.stats[0]),'map50':float(ev.stats[1]),
            'AP_small':float(ev.stats[3]),'AP_medium':float(ev.stats[4]),'AP_large':float(ev.stats[5]),
            'AR_small':float(ev.stats[9]),'AR_medium':float(ev.stats[10]),'AR_large':float(ev.stats[11])}
    result['classes']={}
    for k,category in enumerate(ev.params.catIds):
        values=ev.eval['precision'][:,:,k,0,-1];values=values[values>=0]
        small=ev.eval['precision'][:,:,k,1,-1];small=small[small>=0]
        result['classes'][coco.cats[category]['name']]={'AP':float(values.mean()) if len(values) else None,'AP_small':float(small.mean()) if len(small) else None}
    result['diagnostics_conf025']=diagnose(coco,pred)
    return result,pred

def main():
    root=Path('/root/autodl-tmp/chf_arch_baselines_20261003');out=root/'runs/small_object_audit_20261004';out.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    coco=Path('/root/autodl-tmp/data/chf_arch_rgb_coco_s42');ds=Triples('/root/autodl-tmp/data/train_extracted',coco/'valid/_annotations.coco.json')
    assert len(ds)==400
    loader=DataLoader(ds,batch_size=1,num_workers=4,collate_fn=collate,pin_memory=True)
    cases=[('RGB',root/'runs/rfdetr_m_highres_864x1536_20261003/checkpoint_best_ema.pth',StaticTriModel),
           ('CHF_v2',Path('/root/autodl-tmp/weights/CHF_v2/CHF_v2.pth'),StaticTriModel),
           ('dynamic',root/'runs/rfdetr_dynamic_fusion6_20261004/best.pth',DynamicTriModel)]
    results={}
    for tag,path,cls in cases:
        (out/'status.json').write_text(json.dumps({'state':'evaluating','model':tag}))
        detector=RFDETRMedium.from_checkpoint(path,device='cuda')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model=cls(detector,payload.get('chf_channels',[256])).cuda().eval()
        if 'chf_aux' in payload:model.fusion.load_state_dict(payload['chf_aux'])
        else:model.use_ir=False;model.use_depth=False
        cfg=TrainConfig(dataset_dir=str(coco),output_dir=str(out),batch_size=1)
        _,post=build_criterion_from_config(detector.model_config,cfg)
        results[tag],pred=evaluate(model,post,loader,ds.coco)
        (out/(tag+'_predictions.json')).write_text(json.dumps(pred))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print('CHF AUDIT',tag,results[tag],flush=True)
        model.handle.remove();del model,detector,payload;torch.cuda.empty_cache()
    (out/'status.json').write_text(json.dumps({'state':'complete'}))
if __name__=='__main__':main()
