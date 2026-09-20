import copy
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MM))
from config import default_config
from model import MMYOLO, save_mm_checkpoint, load_mm_checkpoint
from train import (build_optimizer, set_encoder_frozen, apply_bn_policy, make_targets,
                   subset_detection_batch)
from data import MMDataset, AugCfg, collate, scheduled_aug
from independent_fusion import warp, resize_flow, LocalCorrespondence
from ultralytics.utils.loss import v8DetectionLoss


def config(checkpoint=False):
    c = default_config()
    c.fusion.architecture = "independent_p2_memory_v3"
    c.fusion.bus_dim = 128
    c.encoder.share_tier = "a"
    c.encoder.metric_branch = True
    c.encoder.checkpoint_encoder = checkpoint
    c.depth_resampling = "nearest_valid_v2"
    return c


class IndependentV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(4)
        rgb = torch.rand(2,3,64,96)
        depth = torch.rand(2,4,64,96)
        depth[:,2:] = 1
        return rgb, rgb[:,:1].clone(),depth

    def test_independence_pretraining_optimizer_and_freeze(self):
        m = MMYOLO(config())
        rgb,ir,dep = m.encoder_modules()
        sets = [{p.data_ptr() for p in module.parameters()} for module in (rgb,ir,dep)]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertTrue(torch.equal(rgb[0].conv.weight.sum(1,keepdim=True),ir[0].conv.weight))
        self.assertTrue(torch.equal(rgb[6].state_dict()["cv1.conv.weight"],dep[6].state_dict()["cv1.conv.weight"]))
        opt = build_optimizer(m,.0004,.5)
        ids = {id(p) for g in opt.param_groups if g["role"]=="encoder" for p in g["params"]}
        self.assertTrue(all(id(p) in ids for p in dep.parameters()))
        set_encoder_frozen(m,True)
        apply_bn_policy(m,"adaptive",True)
        self.assertFalse(ir[0].bn.training)
        self.assertFalse(any(p.requires_grad for p in ir.parameters()))
        set_encoder_frozen(m,False)
        self.assertTrue(all(p.requires_grad for p in ir.parameters()))

    def test_real_loss_gradients_checkpoint_bn_and_p2(self):
        m = MMYOLO(config(checkpoint=True)).train()
        rgb,ir,dep = self.inputs()
        before = int(m.aux_encoders["dep"][0].bn.num_batches_tracked)
        out = m(rgb,ir,dep)
        self.assertEqual([tuple(x.shape[-2:]) for x in out["feats"]],[(16,24),(8,12),(4,6),(2,3)])
        batch = {"boxes":[torch.tensor([[0.,.5,.5,.2,.2]])]*2}
        loss,_ = v8DetectionLoss(m)(out,make_targets(batch,(64,96),torch.device("cpu")))
        (loss.sum()/2+m.aux_loss).backward()
        self.assertEqual(int(m.aux_encoders["dep"][0].bn.num_batches_tracked)-before,1)
        for branch in (m.aux_encoders["ir"],m.aux_encoders["dep"],m.metric_encoder,m.embeddings,m.register_bus,m.p2_neck):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_masking_memory_reset_and_strict_roundtrip(self):
        m = MMYOLO(config()).eval()
        rgb,ir,dep = self.inputs()
        keep = {"rgb":torch.ones(2),"ir":torch.ones(2),"dep":torch.zeros(2)}
        with torch.no_grad():
            a = m(rgb,ir,dep,keep=keep)[0]
            b = m(rgb,ir,dep*5,keep=keep)[0]
            self.assertTrue(torch.equal(a,b))
            c = m(rgb,ir,dep)[0]
            d = m(rgb,ir,dep)[0]
            self.assertTrue(torch.equal(c,d))
        # Training diagnostics are detached/cleared before deepcopy in this test.
        m.aux_loss = torch.tensor(0.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            save_mm_checkpoint(path,m,meta={"modalities":["rgb","ir","dep"],"canvas":[64,96]})
            restored,_ = load_mm_checkpoint(path)
            with torch.no_grad():
                self.assertTrue(torch.equal(c,restored.eval()(rgb,ir,dep)[0]))

    def test_mosaic_depth_units_scene_ids_masks_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("visible","infrared","depth"):
                (root/m).mkdir()
            samples = []
            for i in range(4):
                name = f"{i}.png"
                for m in ("visible","infrared"):
                    cv2.imwrite(str(root/m/name),np.full((48,80,3),90+i,np.uint8))
                cv2.imwrite(str(root/"depth"/name),np.full((24,40),1000*(i+1),np.uint16))
                samples.append({"stem":str(i),"files":{m:name for m in ("visible","infrared","depth")},"boxes":np.array([[0,.5,.5,.25,.25]],np.float32)})
            aug = AugCfg(mosaic_p=1,scale_range=(1,1),translate=0,misalign_px=0,degrade_p=0,
                         rgb_drop_p=0,aux_drop_p=0,depth_resampling="nearest_valid_v2",total_epochs=10,close_aug_frac=.2)
            ds = MMDataset(root,samples,imgsz=(64,96),aug=aug)
            a,b = ds[(0,0,1)],ds[(0,0,1)]
            self.assertTrue(torch.equal(a["rgb"],b["rgb"]))
            self.assertEqual(set(a["quality"]["scene_id"].unique().tolist()),{1.,2.,3.,4.})
            values = a["depth"][1][a["depth"][2]>0].unique().numpy()
            self.assertTrue(all(np.isclose(v,np.array([.05,.1,.15,.2])).any() for v in values))
            batch = collate([a,b])
            self.assertEqual(batch["quality"]["availability"].shape,(2,3,64,96))
            self.assertEqual(batch["quality"]["scene_id"].shape,(2,1,64,96))
            self.assertEqual(scheduled_aug(aug,8).mosaic_p,0.)

    def test_warp_direction_and_flow_scaling(self):
        x = torch.arange(6.).view(1,1,1,6).expand(1,1,4,6)
        flow = torch.zeros(1,2,4,6)
        flow[:,0] = 1
        self.assertTrue(torch.allclose(warp(x,flow)[...,:-1],x[...,1:],atol=1e-6))
        self.assertTrue(torch.allclose(resize_flow(flow,(8,12))[:,0],torch.full((1,8,12),2.)))

    def test_all_invalid_correspondence_has_finite_zero_gradients(self):
        match = LocalCorrespondence()
        query = torch.randn(2,32,8,12,requires_grad=True)
        key = torch.randn_like(query,requires_grad=True)
        valid = torch.zeros(2,1,8,12)
        flow,conf = match(query,key,torch.ones_like(valid),valid)
        (flow.square().mean()+conf.sum()).backward()
        self.assertEqual(float(conf.sum()),0.)
        self.assertTrue(torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all())

    def test_semantic_heads_flow_and_object_nce_backward(self):
        c = config()
        c.fusion.branch_aux_weight = .15
        c.fusion.flow_supervision_weight = .05
        c.fusion.cross_modal_nce_weight = .03
        c.fusion.p2_match_refine = True
        m = MMYOLO(c).train()
        m.infer_canvas = (64,96)
        rgb,ir,dep = self.inputs()
        out = m(rgb,ir,dep)
        batch = {"boxes":[torch.tensor([[0.,.5,.5,.2,.2]]),
                          torch.tensor([[1.,.4,.4,.3,.2]])]}
        targets = make_targets(batch,(64,96),torch.device("cpu"))
        semantic = m.semantic_regularization(
            targets, torch.tensor([[4.,0.],[0.,-4.]]), torch.ones(2))
        self.assertTrue(torch.isfinite(semantic["flow"]) and torch.isfinite(semantic["nce"]))
        active = m.semantic_branch_present[:,1]
        pred,tgt = subset_detection_batch(m.semantic_branch_predictions["ir"],targets,active)
        branch,_ = v8DetectionLoss(m)(pred,tgt)
        total = out["scores"].sum()*0 + branch.sum()/2 + semantic["flow"] + semantic["nce"]
        total.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.semantic_detect.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.matchers["p2"].parameters()))


if __name__ == "__main__":
    unittest.main()
