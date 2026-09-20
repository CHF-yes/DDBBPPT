"""Portable single-recipe v3 runner; dev selection then explicit all-data refit.

Does not modify train_rgb.py, models_config.py or common/trainer.py. Old ablation
queues are not used. Child failures stop the chain and preserve the last checkpoint.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",default=str(ROOT/"configs/mm_v3_server.json"))
    p.add_argument("--root",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--out",default=str(ROOT/"runs"))
    p.add_argument("--split-file",default=str(ROOT/"configs/split_s42.json"))
    p.add_argument("--name",default="")
    p.add_argument("--batch",type=int)
    p.add_argument("--accum",type=int)
    p.add_argument("--workers",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    for key, value in {
        "dropout_start_epoch": 5,
        "branch_aux_weight": 0.0,
        "flow_supervision_weight": 0.0,
        "cross_modal_nce_weight": 0.0,
        "nce_temperature": 0.10,
        "p2_match_refine": False,
        "run_full_refit": True,
    }.items():
        cfg.setdefault(key, value)
    for key in ("batch","accum","workers"):
        if getattr(a,key) is not None:
            cfg[key] = getattr(a,key)
    name = a.name or ("_smoke_" if a.smoke else "") + cfg["name"]
    run = Path(a.out).resolve()/name
    state_path = run/"pipeline_status.json"
    if state_path.exists() and not a.resume and not a.dry_run:
        raise FileExistsError(f"existing run: {run}; use --resume or new --name")
    state = json.loads(state_path.read_text(encoding="utf-8")) if a.resume and state_path.exists() else {}
    if state.get("state") == "completed":
        state["returncode"] = 0
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(state,ensure_ascii=False)); return
    env = dict(os.environ,PYTHONUTF8="1",PYTHONIOENCODING="utf-8",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    common = [sys.executable,"-u",str(ROOT/"mm_yolo/train.py"),"--root",str(Path(a.root).resolve()),
              "--labels",str(Path(a.labels).resolve()),"--weights",str(Path(a.weights).resolve()),
              "--out",str(run),"--modalities","all","--share-tier","a","--depth-channels","4",
              "--depth-resampling","nearest_valid_v2","--metric-branch","--register-bus",
              "--depth-scales","all","--sampler","coverage","--rare-extra-frac","0.10",
              "--bn-policy","adaptive_no_tail","--memory-control","bounded_v2","--no-prior",
              "--no-deformable","--val-batch","1","--val-conf","0.001","--save-every","1"]
    keys = ("architecture","imgsz","epochs","batch","accum","workers","precision","lr","backbone_lr_mult",
            "freeze_epochs","warmup","lrf","grad_clip","calibrate_clip_steps","mosaic","close_aug_frac",
            "scale_min","scale_max","translate","target_crop_p","misalign_px","degrade_p","rgb_color_p",
            "ir_noise_p","ir_gain_p","depth_hole_p","rgb_dropout","aux_dropout","dropout_start_epoch",
            "branch_aux_weight","flow_supervision_weight","cross_modal_nce_weight","nce_temperature","seed")
    if cfg["checkpoint_encoder"]:
        common += ["--checkpoint-encoder"]
    stages = ["dev"] if (a.smoke or not cfg.get("run_full_refit", True)) else ["dev","full_refit"]
    for stage in stages:
        if state.get(stage+"_complete"):
            continue
        options = {k:cfg[k] for k in keys}
        extra = []
        if stage == "dev":
            if a.smoke:
                options.update(epochs=2,warmup=0,freeze_epochs=0,calibrate_clip_steps=2)
                extra += ["--limit","24","--val-every","1"]
            else:
                extra += ["--split-file",str(Path(a.split_file).resolve()),"--val-every","1"]
        else:
            options.update(epochs=cfg["refit_epochs"],lr=.00006,freeze_epochs=0,warmup=1,lrf=.1,
                           mosaic=0,close_aug_frac=.5,scale_min=.95,scale_max=1.05,translate=.02,
                           target_crop_p=0,misalign_px=0,degrade_p=.02,depth_hole_p=0,
                           rgb_color_p=.10,ir_noise_p=.02,ir_gain_p=.05,
                           rgb_dropout=0,aux_dropout=0,calibrate_clip_steps=32)
            extra += ["--full-data","--val-every","0"]
        cmd = common + ["--name",stage]
        for k,v in options.items():
            cmd += ["--"+k.replace("_","-"),str(v)]
        if cfg.get("p2_match_refine", False):
            cmd += ["--p2-match-refine"]
        last = run/stage/"weights/last.pt"
        if a.resume and last.exists():
            extra += ["--resume"]
        elif stage == "full_refit":
            extra += ["--init-checkpoint",str(run/"dev/weights/best.pt")]
        cmd += extra
        if a.dry_run:
            print(json.dumps(cmd,ensure_ascii=False)); continue
        run.mkdir(parents=True,exist_ok=True)
        (run/"recipe.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
        state.pop("returncode",None)
        state.update(state="running",stage=stage,pid=os.getpid(),started=time.time(),command=cmd)
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        with (run/"console.log").open("a",encoding="utf-8",buffering=1) as log:
            result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state="failed",returncode=result.returncode)
            state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
            raise SystemExit(result.returncode)
        state[stage+"_complete"] = True
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    if not a.dry_run:
        state.update(state="completed",completed=time.time(),returncode=0)
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
