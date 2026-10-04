"""Portable strict V2/V3 400-image validation; no threshold filtering."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import run_rf_trimodal_chf as v2
import run_rf_dynamic_trimodal_chf as v3

def main():
 p=argparse.ArgumentParser();p.add_argument('--version',choices=['v2','v3'],required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',required=True);p.add_argument('--coco',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 assert not a.report.exists();module=v2 if a.version=='v2' else v3
 ds=module.Triples(a.data,a.coco);assert len(ds)==400
 base=torch.load(a.weights,map_location='cpu',weights_only=False);detector=module.RFDETRMedium.from_checkpoint(a.weights,device='cuda');model=module.TriModel(detector,base['chf_channels']).cuda().eval();model.net.load_state_dict(base['model'],strict=True);model.fusion.load_state_dict(base['chf_aux'],strict=True)
 cfg=module.TrainConfig(dataset_dir=str(a.coco.parent.parent),output_dir=str(a.report.parent),batch_size=1);_,post=module.build_criterion_from_config(detector.model_config,cfg)
 result=module.evaluate(model,post,DataLoader(ds,batch_size=1,num_workers=4,collate_fn=module.collate),a.coco);result['sha256']=hashlib.sha256(a.weights.read_bytes()).hexdigest();a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
