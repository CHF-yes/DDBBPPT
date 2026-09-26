"""Short correctness/memory check, not a score experiment."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"mm_yolo"))
from config import default_config
from model import MMYOLO
from train import build_optimizer, make_targets, ensure_finite_state, apply_bn_policy
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA

p = argparse.ArgumentParser()
p.add_argument("--weights",required=True)
p.add_argument("--batch",type=int,default=2)
p.add_argument("--height",type=int,default=736)
p.add_argument("--width",type=int,default=1280)
p.add_argument("--steps",type=int,default=3)
p.add_argument("--checkpoint-encoder",action=argparse.BooleanOptionalAction,default=True)
p.add_argument("--out",required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
c = default_config(weights=a.weights)
c.fusion.architecture="independent_p2_memory_v3"
c.fusion.bus_dim=128
c.encoder.share_tier="a"
c.encoder.metric_branch=True
c.encoder.checkpoint_encoder=a.checkpoint_encoder
c.depth_resampling="nearest_valid_v2"
m = MMYOLO(c).cuda().train()
apply_bn_policy(m,"adaptive_no_tail")
ema = ModelEMA(m)
opt = build_optimizer(m,.0001,.5,wd=.000125)
criterion = v8DetectionLoss(m)
rgb = torch.rand(a.batch,3,a.height,a.width,device="cuda")
ir = torch.rand(a.batch,1,a.height,a.width,device="cuda")
dep = torch.rand(a.batch,4,a.height,a.width,device="cuda")
dep[:,2:] = 1
boxes = torch.tensor([[0.,.5,.5,.2,.2],[5.,.25,.25,.035,.04],[6.,.7,.4,.15,.1]])
target = make_targets({"boxes":[boxes]*a.batch},(a.height,a.width),torch.device("cuda"))
records=[]
for i in range(a.steps):
    torch.cuda.synchronize(); start=time.time()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        pred=m(rgb,ir,dep)
        loss,items=criterion(pred,target)
        total=loss.sum()/a.batch+m.aux_loss
    assert torch.isfinite(total)
    total.backward()
    grad=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in m.parameters() if p.grad is not None]))
    assert torch.isfinite(grad)
    for module in (m.aux_encoders["ir"],m.aux_encoders["dep"],m.metric_encoder,m.register_bus,m.p2_neck):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    opt.step(); ema.update(m)
    torch.cuda.synchronize()
    records.append({"step":i,"loss":float(total),"grad_norm":float(grad),"seconds":time.time()-start,
                    "allocated_gib":torch.cuda.max_memory_allocated()/2**30,"reserved_gib":torch.cuda.max_memory_reserved()/2**30})
    print(json.dumps(records[-1]),flush=True)
ensure_finite_state(m)
ensure_finite_state(ema.ema)
with torch.no_grad():
    prediction=ema.ema.eval()(rgb[:1],ir[:1],dep[:1])[0]
assert prediction.shape[1]==16 and torch.isfinite(prediction).all()
report={"parameters":m.param_report(),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"canvas":[a.height,a.width],"batch":a.batch,"checkpoint_encoder":a.checkpoint_encoder,"records":records,"passed":True}
Path(a.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
print("PREFLIGHT_OK",flush=True)
