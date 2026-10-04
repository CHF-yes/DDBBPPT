"""Original CHF_v3, identical validation/test preprocessing; original-normalized TXT."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from run_rf_dynamic_trimodal_chf import Triples,collate,TriModel,RFDETRMedium,TrainConfig,build_criterion_from_config
from audit_rf_small_objects_chf import evaluate

@torch.no_grad()
def main():
 torch.set_num_threads(8)
 root=Path('/root/autodl-tmp/chf_arch_baselines_20261003');out=root/'deliverables/CHF_v3_test1000_20261004';out.mkdir(exist_ok=False)
 weights=Path('/root/autodl-tmp/weights/CHF_v3/CHF_v3.pth');sha=hashlib.sha256(weights.read_bytes()).hexdigest();assert sha=='936755d6f850ce7b19b12deb27b3af52df804ef4559f93cdabe1719b8ca7e977'
 payload=torch.load(weights,map_location='cpu',weights_only=False);detector=RFDETRMedium.from_checkpoint(weights,device='cuda');model=TriModel(detector,payload['chf_channels']).cuda().eval();model.net.load_state_dict(payload['model'],strict=True);model.fusion.load_state_dict(payload['chf_aux'],strict=True)
 coco=Path('/root/autodl-tmp/data/chf_arch_rgb_coco_s42');_,post=build_criterion_from_config(detector.model_config,TrainConfig(dataset_dir=str(coco),output_dir=str(out),batch_size=1))
 (out/'status.json').write_text(json.dumps({'state':'validation_check'}))
 valid=Triples('/root/autodl-tmp/data/train_extracted',coco/'valid/_annotations.coco.json');valresult,_=evaluate(model,post,DataLoader(valid,batch_size=1,num_workers=4,collate_fn=collate),valid.coco)
 assert abs(valresult['map50_95']-.5103743165)<.0001,valresult
 (out/'validation_check.json').write_text(json.dumps(valresult,indent=2));print('VALIDATION CHECK',valresult['map50_95'],flush=True)
 data=Path('/root/autodl-tmp/data/test_p2_extracted');files=sorted(p for p in (data/'visible').iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']);assert len(files)==1000
 names=[p.stem for p in files];assert len(set(names))==1000
 for mod in ['infrared','depth']:assert {p.name for p in (data/mod).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']}=={p.name for p in files}
 meta={'images':[],'annotations':[],'categories':[{'id':i,'name':str(i)} for i in range(12)]}
 for index,p in enumerate(files):
  with Image.open(p) as im:w,h=im.size
  meta['images'].append({'id':index+1,'file_name':p.name,'width':w,'height':h})
 cp=out/'test_images.json';cp.write_text(json.dumps(meta));ds=Triples(data,cp);labels=out/'labels';labels.mkdir();start=time.time();rejected=[]
 for step,(r,i,d,t) in enumerate(DataLoader(ds,batch_size=1,num_workers=4,collate_fn=collate)):
  result=post(model(r.cuda(),i.cuda(),d.cuda()),torch.stack([x['orig_size'] for x in t]).cuda())[0];h,w=map(int,t[0]['orig_size']);rows=[]
  for j in result['scores'].argsort(descending=True).tolist():
   c=int(result['labels'][j]);score=float(result['scores'][j]);box=result['boxes'][j].float().cpu().numpy();box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
   if not (0<=c<12 and np.isfinite(box).all() and np.isfinite(score) and 0<=score<=1):
    rejected.append({'image':files[step].name,'class':c,'score':str(score),'box':list(map(str,box))});continue
   x1,y1,x2,y2=map(float,box)
   if x2<=x1 or y2<=y1:continue
   rows.append(f'{c} {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f} {score:.9f}')
   if len(rows)==100:break
  (labels/(files[step].stem+'.txt')).write_text('\n'.join(rows)+('\n' if rows else ''))
  if step%100==0:
   (out/'status.json').write_text(json.dumps({'state':'inference','completed':step+1,'total':1000,'elapsed':time.time()-start}));print('TEST',step+1,flush=True)
 assert {p.stem for p in labels.glob('*.txt')}==set(names)
 counts=[]
 for p in labels.glob('*.txt'):
  rows=p.read_text().splitlines();assert len(rows)<=100;counts.append(len(rows))
  for row in rows:
   vals=list(map(float,row.split()));assert len(vals)==6 and vals[0].is_integer() and 0<=vals[0]<12 and all(np.isfinite(v) and 0<=v<=1 for v in vals[1:])
 zip_path=out.parent/'CHF_v3_test1000_labels.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(labels.glob('*.txt')):z.write(p,p.name)
 report={'state':'complete','weight_sha256':sha,'model':'original CHF_v3','coordinate_space':'original image normalized cxcywh','format':'class cx cy w h confidence','files':1000,'max_boxes':max(counts),'empty_files':counts.count(0),'rejected_candidates':rejected,'validation_AP95':valresult['map50_95'],'zip':str(zip_path),'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'elapsed':time.time()-start}
 (out/'status.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report,flush=True)
if __name__=='__main__':main()
