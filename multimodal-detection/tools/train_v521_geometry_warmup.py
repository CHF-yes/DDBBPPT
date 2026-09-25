"""Warm up V5.2.1 global geometry before IR detector training.

This prevents the independent detector from learning on heavily distorted IR
while the large residual head is still near identity.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'vendor', ROOT / 'mm_yolo'):
    sys.path.insert(0, str(path))

from config import MMConfig
from data import AugCfg, MMDataset, build_index, collate, load_split
from model import MMYOLO, save_mm_checkpoint
from train import adapt_depth_checkpoint_state
from v52_stage_a import resample
from v521_stage_a import residual_sampling


def make_aug(cache, probability):
    return AugCfg(
        imgsz=(736, 1280), scale_range=(1., 1.), translate=0.,
        hflip_p=0., ir_read_mode='median_channel', ir_a0_cache=str(cache),
        require_ir_a0=True, depth_resampling='nearest_valid_v2',
        misalign_px=0, mosaic_p=0, rgb_drop_p=0, aux_drop_p=0,
        degrade_p=0, rgb_color_p=0, ir_noise_p=0, ir_gain_p=0,
        target_crop_p=0, ir_affine_p=probability, v521_explicit=True,
        v521_angle_deg=15., v521_shift_frac=.20,
        v521_scale_min=.80, v521_scale_max=1.25)


def aligned_ir_reference(model, ir, normalized_target, quality):
    """Reconstruct the pre-perturbation IR used as an exact warm-up reference.

    Real RGB/IR pairs contain unknown base misalignment, so supervising only the
    added synthetic motion against RGB gives contradictory labels.  Same-modal
    synthetic registration first teaches the head what rotation/translation/
    scale mean; Stage A can then adapt it to RGB edges and detection losses.
    """
    aligner = model.v52_ir_input
    residual, _ = residual_sampling(
        normalized_target, ir.shape[-2:], aligner.max_angle,
        aligner.max_shift, aligner.max_log_scale)
    sampling = quality['v52_ir_sampling'].float()
    s3 = sampling.new_zeros((len(sampling), 3, 3))
    s3[:, :2] = sampling
    s3[:, 2, 2] = 1
    total = torch.bmm(s3, residual)[:, :2]
    # Exact reference is the synthetically corrected image after applying the
    # same A0 coarse transform that the model receives.  This teaches only the
    # residual motion and never treats an uncertain A0 candidate as truth.
    return resample(ir, total, ir.shape[-2:]).detach()


def forward_geometry(model, batch, device):
    ir = batch['ir'].to(device, non_blocking=True)
    quality = {key: value.to(device, non_blocking=True)
               for key, value in batch['quality'].items()}
    target = batch['ir_affine_target'].to(device, non_blocking=True)
    reference = aligned_ir_reference(model, ir, target, quality)
    present = batch['keep']['ir'].to(device).bool()
    raw = model._encode(ir, 'ir', present)
    model.v52_ir_input(raw, quality, ir.shape[-2:], rgb=reference, ir=ir)
    return model.v52_ir_input.supervision_loss(
        target,
        batch['ir_affine_supervised'].to(device),
        batch['ir_affine_confidence'].to(device))


@torch.no_grad()
def evaluate(model, loader, device, synthetic):
    model.eval()
    predicted, target, confidence = [], [], []
    for batch in loader:
        ir = batch['ir'].to(device)
        quality = {key: value.to(device) for key, value in batch['quality'].items()}
        normalized_target = batch['ir_affine_target'].to(device)
        reference = aligned_ir_reference(model, ir, normalized_target, quality)
        present = batch['keep']['ir'].to(device).bool()
        raw = model._encode(ir, 'ir', present)
        model.v52_ir_input(raw, quality, ir.shape[-2:], rgb=reference, ir=ir)
        predicted.append(model.v52_ir_input.last_global_normalized.float().cpu())
        target.append(batch['ir_affine_target'].float())
        confidence.append(model.v52_ir_input.last_global_logit.float().sigmoid().cpu())
    pred = torch.cat(predicted)
    tgt = torch.cat(target)
    conf = torch.cat(confidence)
    report = {'confidence_mean': float(conf.mean()),
              'residual_abs_mean': float(pred.abs().mean())}
    if synthetic:
        error = (pred - tgt).abs()
        pred_scale = torch.exp(pred[:, 3] * np.log(1.25))
        target_scale = torch.exp(tgt[:, 3] * np.log(1.25))
        report.update(
            angle_mae_deg=float(error[:, 0].mean() * 15.),
            tx_mae_px=float(error[:, 1].mean() * .20 * 1280),
            ty_mae_px=float(error[:, 2].mean() * .20 * 736),
            scale_mae=float((pred_scale - target_scale).abs().mean()),
            normalized_mae=float(error.mean()))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--train-limit', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)
    np.random.seed(42)

    base = Path('/root/autodl-tmp/runs/mm_v44_multimodal_ceiling_rect_s42_b4a4/stage_b/weights/best.pt')
    expected = '0a38f26fcfcf9bc0d909e6aa26f80b6de7d0ae952780d1e2213dbefa625b6ee8'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == expected
    checkpoint = torch.load(base, map_location='cpu', weights_only=False)
    cfg = MMConfig.from_structure(checkpoint['structure'])
    cfg.weights = '/root/autodl-tmp/weights/yolo11s.pt'
    cfg.fusion.fusion_strategy = 'v521_stage_a_v1'
    cfg.fusion.branch_aux_weights = (1., 1., .6)
    cfg.fusion.quality_channels = 10
    cfg.fusion.ir_coarse_align = False
    cfg.encoder.checkpoint_encoder = True
    cfg.ir_read_mode = 'median_channel'
    model = MMYOLO(cfg).to(device)
    state, _ = adapt_depth_checkpoint_state(checkpoint['model_state'], model)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.v52_ir_input.global_features, model.v52_ir_input.global_head,
                   model.v52_ir_input.ghost_refiner):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    root = Path('/root/autodl-tmp/data/train_extracted')
    index = build_index(
        root, Path('/root/autodl-tmp/data/new_labels_2000'),
        exclude_stems=Path('/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt'))
    train_rows, val_rows = load_split(args.cache / 'split_s42_v52.json', index)
    if args.train_limit > 0 and len(train_rows) > args.train_limit:
        order = np.random.default_rng(42).permutation(len(train_rows))[:args.train_limit]
        train_rows = [train_rows[int(i)] for i in order]
    train_ds = MMDataset(root, train_rows, imgsz=(736, 1280), train=True,
                         aug=make_aug(args.cache, .70), enabled=['rgb', 'ir'], seed=42)
    synth_ds = MMDataset(root, val_rows[:160], imgsz=(736, 1280), train=True,
                         aug=make_aug(args.cache, 1.0), enabled=['rgb', 'ir'], seed=314)
    identity_ds = MMDataset(root, val_rows[:160], imgsz=(736, 1280), train=True,
                            aug=make_aug(args.cache, 0.0), enabled=['rgb', 'ir'], seed=2718)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, drop_last=False,
                              pin_memory=True, persistent_workers=args.workers > 0)
    synth_loader = DataLoader(synth_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True,
                              persistent_workers=args.workers > 0)
    identity_loader = DataLoader(identity_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                 persistent_workers=args.workers > 0)

    history, best_score = [], float('inf')
    checkpoint_meta = {
        'stage': 'v521_geometry_warmup',
        'modalities': ['rgb', 'ir', 'dep'],
        'canvas': [736, 1280],
        'imgsz': [736, 1280],
        'name': 'v521_geometry_warmup',
    }
    print(json.dumps({'train_samples': len(train_ds), 'synthetic_eval': len(synth_ds),
                      'identity_eval': len(identity_ds), 'workers': args.workers}),
          flush=True)
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running, steps = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = forward_geometry(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite geometry warmup loss at epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 20.)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if steps % 50 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': steps,
                                  'mean_loss': running / steps}), flush=True)
        synthetic = evaluate(model, synth_loader, device, synthetic=True)
        identity = evaluate(model, identity_loader, device, synthetic=False)
        score = synthetic['normalized_mae'] + max(0., identity['confidence_mean'] - .20)
        row = {'epoch': epoch + 1, 'train_loss': running / max(steps, 1),
               'synthetic': synthetic, 'identity': identity}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_mm_checkpoint(args.out / 'last.pt', model, epoch=epoch + 1,
                           best_map=-score, meta=checkpoint_meta)
        if score < best_score:
            best_score = score
            save_mm_checkpoint(args.out / 'best.pt', model, epoch=epoch + 1,
                               best_map=-score, meta=checkpoint_meta)
            (args.out / 'best_metrics.json').write_text(
                json.dumps(row, indent=2), encoding='utf-8')
    (args.out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'best_score': best_score,
                      'best': json.loads((args.out / 'best_metrics.json').read_text())}, indent=2))


if __name__ == '__main__':
    main()
