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
    p.add_argument("--init-checkpoint",default="",help="dev 阶段用已有权重热启动（保留超参可变）")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    for key, value in {
        "dropout_start_epoch": 5,
        "branch_aux_weight": 0.0,
        "flow_supervision_weight": 0.0,
        "cross_modal_nce_weight": 0.0,
        "embedding_recon_weight": 0.01,
        "embedding_alignment_weight": 0.005,
        "nce_temperature": 0.10,
        "p2_match_refine": False,
        "match_floor": 0.0,
        "branch_aux_weights": [],
        "val_every": 1,
        "bn_policy": "adaptive_no_tail",
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
              "--memory-control","bounded_v2","--no-prior",
              "--no-deformable","--val-batch","1","--val-conf","0.001","--save-every","1"]
    keys = ("architecture","imgsz","epochs","batch","accum","workers","precision","lr","backbone_lr_mult",
            "fusion_lr_mult","p2_lr_mult","detector_lr_mult","semantic_lr_mult","ir_read_mode",
            "freeze_epochs","warmup","lrf","grad_clip","calibrate_clip_steps","bn_policy","mosaic","close_aug_frac",
            "scale_min","scale_max","translate","rotate_deg","target_crop_p","target_occlusion_p",
            "ir_affine_p","ir_affine_deg","ir_affine_shift","ir_affine_scale",
            "misalign_px","degrade_p","rgb_color_p",
            "ir_noise_p","ir_gain_p","depth_hole_p","rgb_dropout","aux_dropout","dropout_start_epoch",
            "branch_aux_weight","flow_supervision_weight","cross_modal_nce_weight","nce_temperature",
            "match_floor","branch_aux_weights","branch_aux_end_weights",
            "flow_supervision_end_weight","cross_modal_nce_end_weight","alignment_mode",
            "depth_reliability","flow_identity_weight","embedding_recon_weight",
            "embedding_recon_end_weight","embedding_alignment_weight",
            "embedding_alignment_end_weight","train_stage","aux_branch_mode",
            "independent_preserve_weight","independent_preserve_end_weight",
            "fusion_strategy","ir_affine_loss_weight","ir_affine_loss_end_weight",
            "evidence_supervision_weight","evidence_supervision_end_weight",
            "val_every","seed")

    def add_options(cmd, options):
        for key, value in options.items():
            if value is None:
                continue
            if key in ("branch_aux_weights", "branch_aux_end_weights"):
                if value:
                    cmd += ["--"+key.replace("_","-")] + [str(x) for x in value]
            else:
                cmd += ["--"+key.replace("_","-"), str(value)]
        return cmd

    if cfg["checkpoint_encoder"]:
        common += ["--checkpoint-encoder"]

    # V4.2 is an explicit two-stage dev pipeline.  Stage A teaches IR/Depth to
    # enter the established detector space with RGB/neck/head frozen.  Stage B
    # warm-starts from Stage A's best validation checkpoint and jointly fine-tunes
    # at a low LR.  Stage A can regress after its peak, so last.pt is not a safe
    # hand-off checkpoint for model selection pipelines.
    if cfg.get("pipeline") == "v42_two_stage":
        phases = ["stage_a", "stage_b"]
        if not a.init_checkpoint and not a.resume:
            raise ValueError("V4.2 首次启动必须提供 --init-checkpoint")
        for phase_name in phases:
            if state.get(phase_name+"_complete"):
                continue
            phase = dict(cfg)
            phase.update(cfg[phase_name])
            if a.smoke:
                phase.update(epochs=1, warmup=0, freeze_epochs=0,
                             calibrate_clip_steps=1, val_every=1)
            options = {k:phase[k] for k in keys if k in phase}
            cmd = add_options(common + ["--name",phase_name], options)
            if phase.get("p2_match_refine", False):
                cmd += ["--p2-match-refine"]
            if phase.get("ir_coarse_align", False):
                cmd += ["--ir-coarse-align"]
            if phase.get("preserve_init_fusion", False):
                cmd += ["--preserve-init-fusion"]
            if phase.get("eval_initial", True):
                cmd += ["--eval-initial"]
            if a.smoke:
                cmd += ["--limit","24"]
            else:
                cmd += ["--split-file",str(Path(a.split_file).resolve())]
            last = run/phase_name/"weights/last.pt"
            if a.resume and last.exists():
                cmd += ["--resume"]
            elif phase_name == "stage_a":
                if not a.init_checkpoint:
                    raise FileNotFoundError("Stage A 没有 checkpoint 可恢复，也未提供 --init-checkpoint")
                cmd += ["--init-checkpoint",str(Path(a.init_checkpoint).resolve())]
            else:
                source = run/"stage_a"/"weights"/"best.pt"
                if not a.dry_run and not source.is_file():
                    raise FileNotFoundError(f"Stage B 缺少 Stage A best.pt: {source}")
                cmd += ["--init-checkpoint",str(source)]
            if a.dry_run:
                print(json.dumps(cmd,ensure_ascii=False)); continue
            run.mkdir(parents=True,exist_ok=True)
            (run/"recipe.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
            state.pop("returncode",None)
            state.update(state="running",stage=phase_name,pid=os.getpid(),started=time.time(),command=cmd)
            state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
            with (run/"console.log").open("a",encoding="utf-8",buffering=1) as log:
                result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(state="failed",returncode=result.returncode)
                state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
                raise SystemExit(result.returncode)
            state[phase_name+"_complete"] = True
            state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        if not a.dry_run:
            state.update(state="completed",completed=time.time(),returncode=0)
            state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps(state,ensure_ascii=False),flush=True)
        return

    stages = ["dev"] if (a.smoke or not cfg.get("run_full_refit", True)) else ["dev","full_refit"]
    for stage in stages:
        if state.get(stage+"_complete"):
            continue
        options = {k:cfg[k] for k in keys if k in cfg}
        extra = []
        if stage == "dev":
            if a.smoke:
                options.update(epochs=2,warmup=0,freeze_epochs=0,calibrate_clip_steps=2,val_every=1)
                extra += ["--limit","24"]
            else:
                extra += ["--split-file",str(Path(a.split_file).resolve())]
            if a.init_checkpoint:
                extra += ["--init-checkpoint",str(Path(a.init_checkpoint).resolve())]
        else:
            options.update(epochs=cfg["refit_epochs"],lr=.00006,freeze_epochs=0,warmup=1,lrf=.1,
                           mosaic=0,close_aug_frac=.5,scale_min=.95,scale_max=1.05,translate=.02,
                           target_crop_p=0,misalign_px=0,degrade_p=.02,depth_hole_p=0,
                           rgb_color_p=.10,ir_noise_p=.02,ir_gain_p=.05,
                           rgb_dropout=0,aux_dropout=0,calibrate_clip_steps=32,val_every=0)
            extra += ["--full-data"]
        cmd = common + ["--name",stage]
        cmd = add_options(cmd, options)
        if cfg.get("p2_match_refine", False):
            cmd += ["--p2-match-refine"]
        if cfg.get("ir_coarse_align", False):
            cmd += ["--ir-coarse-align"]
        if cfg.get("preserve_init_fusion", False):
            cmd += ["--preserve-init-fusion"]
        if cfg.get("eval_initial", False) and stage == "dev":
            cmd += ["--eval-initial"]
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
