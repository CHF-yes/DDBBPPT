"""Real GPU optimizer-step, modality isolation and geometry checks before launch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'vendor',ROOT/'mm_yolo'): sys.path.insert(0,str(p))
import cv2
import numpy as np
import torch
from config import MMConfig
from model import MMYOLO, save_mm_checkpoint, load_mm_checkpoint
from train import (adapt_depth_checkpoint_state, set_independent_aux_mode,
                   set_frozen_bn_eval, build_optimizer, make_targets, subset_detection_batch,
                   v8DetectionLoss, validate_checkpoint)
from data import build_index, load_split, AugCfg, MMDataset, collate, _mat3
from v52_stage_a import resample


def main():
    a=argparse.ArgumentParser(); a.add_argument('--out',required=True); a=a.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42); dev='cuda:0'
    report={}; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    base=Path('/root/autodl-tmp/runs/mm_v44_multimodal_ceiling_rect_s42_b4a4/stage_b/weights/best.pt')
    assert hashlib.sha256(base.read_bytes()).hexdigest()=='0a38f26fcfcf9bc0d909e6aa26f80b6de7d0ae952780d1e2213dbefa625b6ee8'
    ck=torch.load(base,map_location='cpu',weights_only=False)
    cfg=MMConfig.from_structure(ck['structure']); cfg.weights='/root/autodl-tmp/weights/yolo11s.pt'
    cfg.fusion.fusion_strategy='v52_stage_a_v1'; cfg.fusion.branch_aux_weights=(1.,1.,.6)
    cfg.fusion.quality_channels=10; cfg.fusion.ir_coarse_align=False
    cfg.encoder.checkpoint_encoder=True; cfg.ir_read_mode='median_channel'
    model=MMYOLO(cfg).to(dev)
    state,migrated=adapt_depth_checkpoint_state(ck['model_state'],model)
    model.load_state_dict(state,strict=True)
    assert len(model.evidence_router)==0
    assert set(model.independent_aux)=={'rgb','ir','dep'}
    for k,v in ck['model_state'].items():
        assert torch.equal(model.state_dict()[k].cpu(),v.cpu()), k
    report['v44_existing_tensors_exact']=True
    root=Path('/root/autodl-tmp/data/train_extracted'); cache=Path('/root/autodl-tmp/cache/ir_a0_v52_rotation_input_v1')
    idx=build_index(root,Path('/root/autodl-tmp/data/new_labels_2000'),
                    exclude_stems=Path('/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt'))
    tr,va=load_split(cache/'split_s42_v52.json',idx); assert (len(tr),len(va))==(1598,400)
    report['data']={'train':len(tr),'val':len(va)}
    selected=[next(s for s in tr if s['stem']==st) for st in ['001743','000037']]
    aug=AugCfg(imgsz=(736,1280), ir_read_mode='median_channel',ir_a0_cache=str(cache),
               require_ir_a0=True,depth_resampling='nearest_valid_v2',misalign_px=0,
               rgb_drop_p=0,aux_drop_p=0,scale_range=(.92,1.08),translate=.025)
    ds=MMDataset(root,selected,imgsz=(736,1280),train=True,aug=aug,enabled=['rgb','ir','dep'])
    items=[ds[i] for i in range(2)]; batch=collate(items)
    # Verify conjugation under actual random training crop/flip/letterbox.
    for item in items:
        with np.load(cache/'samples'/f"{item['stem']}.npz") as z:
            C=_mat3(z['v52_coarse_matrix']); assert z['affine_supervised']==0
        M=_mat3(item['M'].numpy()); S=_mat3(item['quality']['v52_ir_sampling'].numpy())
        assert np.allclose(S@M@C,M,atol=.001)
    report['crop_flip_rotation_conjugation']=True
    # Independent OpenCV check for inverse sampling convention and pixel centers.
    signal=np.zeros((96,160),np.float32); signal[21:32,101:112]=1
    C=cv2.getRotationMatrix2D((79.5,47.5),17,1)
    expected=cv2.warpAffine(signal,C,(160,96))
    actual=resample(torch.tensor(signal)[None,None].cuda(),
                    torch.tensor(cv2.invertAffineTransform(C))[None].cuda(),(96,160)).cpu().numpy()[0,0]
    assert np.abs(expected-actual).mean()<.001
    report['opencv_rotation_mae']=float(np.abs(expected-actual).mean())
    tensors={k:batch[k].to(dev) for k in ['rgb','ir','depth']}
    quality={k:v.to(dev) for k,v in batch['quality'].items()}
    keep={k:v.to(dev) for k,v in batch['keep'].items()}
    # IR predictions must not change when RGB pixels and RGB quality change,
    # given the same permitted offline geometric calibration.
    model.eval()
    with torch.no_grad():
        before,_=model.independent_branch_prediction('ir',**tensors,keep=keep,quality=quality)
        changed={**tensors,'rgb':torch.rand_like(tensors['rgb'])}
        changedq={**quality,'rgb':torch.rand_like(quality['rgb'])}
        after,_=model.independent_branch_prediction('ir',**changed,keep=keep,quality=changedq)
        assert torch.equal(before[0],after[0])
    report['ir_rgb_semantic_isolation']=True
    del before,after
    model.train(); set_independent_aux_mode(model); set_frozen_bn_eval(model)
    rgb_stem=model.backbone.model[0].conv.weight.detach().clone()
    ir_stem=model.aux_encoders['ir'][0].conv.weight.detach().clone()
    dep_stem=model.aux_encoders['dep'][0].conv.weight.detach().clone()
    opt=build_optimizer(model,2e-4,.25); crit=v8DetectionLoss(model)
    target=make_targets(batch,(736,1280),torch.device(dev)); losses={}
    for name in ('rgb','ir','dep'):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred,active=model.independent_branch_prediction(name,**tensors,keep=keep,quality=quality)
            pred,target_sub=subset_detection_batch(pred,target,active)
            vec,_=crit(pred,target_sub); loss=vec.sum()/2
            if name=='ir': loss=loss+.01*model.v52_ir_input.last_penalty
        assert torch.isfinite(loss); loss.backward(); losses[name]=float(loss.detach())
        del pred,loss
    gradients={name:float(module.weight.grad.norm()) for name,module in
               [('ir',model.aux_encoders['ir'][0].conv),('dep',model.aux_encoders['dep'][0].conv),
                ('residual',model.v52_ir_input.offset[-1])]}
    assert all(np.isfinite(v) and v>0 for v in gradients.values())
    torch.nn.utils.clip_grad_norm_(model.parameters(),60); opt.step()
    assert torch.equal(rgb_stem,model.backbone.model[0].conv.weight)
    assert not torch.equal(ir_stem,model.aux_encoders['ir'][0].conv.weight)
    assert not torch.equal(dep_stem,model.aux_encoders['dep'][0].conv.weight)
    report.update(losses=losses,gradients=gradients,rgb_frozen=True,ir_depth_updated=True,
                  gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3)
    save_mm_checkpoint(out/'roundtrip.pt',model,epoch=1)
    restored,_=load_mm_checkpoint(out/'roundtrip.pt',device='cpu',weights=cfg.weights)
    assert restored.cfg.fusion.fusion_strategy=='v52_stage_a_v1'
    report['checkpoint_roundtrip']=True; report['passed']=True
    (out/'preflight.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
