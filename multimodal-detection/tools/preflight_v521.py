"""GPU preflight for V5.2.1 explicit coarse Thermal/Ghost Stage A."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'vendor', ROOT / 'mm_yolo'):
    sys.path.insert(0, str(path))

import numpy as np
import torch

from config import MMConfig
from data import AugCfg, MMDataset, build_index, collate, load_split
from model import MMYOLO, load_mm_checkpoint, save_mm_checkpoint
from train import (adapt_depth_checkpoint_state, build_optimizer, make_targets,
                   set_frozen_bn_eval, set_independent_aux_mode,
                   subset_detection_batch, v8DetectionLoss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    report = {}

    base = Path('/root/autodl-tmp/runs/mm_v44_multimodal_ceiling_rect_s42_b4a4/stage_b/weights/best.pt')
    expected = '0a38f26fcfcf9bc0d909e6aa26f80b6de7d0ae952780d1e2213dbefa625b6ee8'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == expected
    checkpoint = torch.load(base, map_location='cpu', weights_only=False)
    cfg = MMConfig.from_structure(checkpoint['structure'])
    cfg.weights = '/root/autodl-tmp/weights/yolo11s.pt'
    cfg.fusion.fusion_strategy = 'v521_stage_a_v1'
    cfg.fusion.branch_aux_weights = (1., 1., .6)
    cfg.fusion.branch_aux_weight = 0.
    cfg.fusion.quality_channels = 10
    cfg.fusion.ir_coarse_align = False
    cfg.encoder.checkpoint_encoder = True
    cfg.ir_read_mode = 'median_channel'
    model = MMYOLO(cfg).to(device)
    state, migrated = adapt_depth_checkpoint_state(checkpoint['model_state'], model)
    model.load_state_dict(state, strict=True)
    report['migration'] = migrated
    report['v44_sha256'] = expected

    root = Path('/root/autodl-tmp/data/train_extracted')
    index = build_index(
        root, Path('/root/autodl-tmp/data/new_labels_2000'),
        exclude_stems=Path('/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt'))
    train_rows, val_rows = load_split(args.cache / 'split_s42_v52.json', index)
    assert (len(train_rows), len(val_rows)) == (1598, 400)
    chosen = []
    for row in train_rows:
        with np.load(args.cache / 'samples' / f"{row['stem']}.npz", allow_pickle=False) as z:
            if float(np.asarray(z['ghost_probability']).max()) > .005:
                chosen.append(row)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2
    aug = AugCfg(
        imgsz=(736, 1280), scale_range=(.92, 1.08), translate=.025,
        ir_read_mode='median_channel', ir_a0_cache=str(args.cache),
        require_ir_a0=True, depth_resampling='nearest_valid_v2',
        misalign_px=0, mosaic_p=0, rgb_drop_p=0, aux_drop_p=0,
        ir_affine_p=1.0, v521_explicit=True, v521_angle_deg=15.,
        v521_shift_frac=.20, v521_scale_min=.80, v521_scale_max=1.25)
    dataset = MMDataset(root, chosen, imgsz=(736, 1280), train=True, aug=aug,
                        enabled=['rgb', 'ir', 'dep'], seed=42)
    items = [dataset[i] for i in range(2)]
    batch = collate(items)
    required = {'v521_ghost_probability', 'v521_ghost_confidence',
                'v521_coarse_available', 'v521_thermal_confidence',
                'v521_hard_mask', 'v521_soft_mask', 'v521_align_exclude',
                'v52_ir_sampling'}
    assert required.issubset(batch['quality'])
    assert batch['ir_affine_supervised'].eq(1).all()
    assert batch['ir_affine_target'].abs().max() <= 1.001
    report['samples'] = batch['stems']
    report['explicit_inputs'] = {
        key: {'shape': list(batch['quality'][key].shape),
              'mean': float(batch['quality'][key].float().mean()),
              'max': float(batch['quality'][key].float().max())}
        for key in sorted(required) if key != 'v52_ir_sampling'
    }
    report['synthetic_target'] = batch['ir_affine_target'].tolist()

    tensors = {key: batch[key].to(device) for key in ('rgb', 'ir', 'depth')}
    quality = {key: value.to(device) for key, value in batch['quality'].items()}
    keep = {key: value.to(device) for key, value in batch['keep'].items()}
    target = make_targets(batch, (736, 1280), device)
    set_independent_aux_mode(model)
    set_frozen_bn_eval(model)
    model.train()
    optimizer = build_optimizer(model, 2e-4, .25)
    criterion = v8DetectionLoss(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction, active = model.independent_branch_prediction(
            'ir', **tensors, keep=keep, quality=quality)
        prediction, target_ir = subset_detection_batch(prediction, target, active)
        detection_vector, _ = criterion(prediction, target_ir)
        detection_loss = detection_vector.sum() / 2
        geometry_loss = model.v52_ir_input.supervision_loss(
            batch['ir_affine_target'].to(device),
            batch['ir_affine_supervised'].to(device),
            batch['ir_affine_confidence'].to(device))
        total = detection_loss + .5 * geometry_loss + .01 * model.v52_ir_input.last_penalty
    assert torch.isfinite(total)
    total.backward()
    gradients = {
        'ir_encoder': float(model.aux_encoders['ir'][0].conv.weight.grad.float().norm()),
        'global_head': float(model.v52_ir_input.global_head[-1].weight.grad.float().norm()),
        'local_head': float(model.v52_ir_input.local_head[-1].weight.grad.float().norm()),
    }
    assert all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients
    assert model.backbone.model[0].conv.weight.grad is None
    optimizer.step()
    report.update(
        detection_loss=float(detection_loss.detach()),
        geometry_loss=float(geometry_loss.detach()), gradients=gradients,
        rgb_backbone_gradient=False,
        geometry_stats=model.v52_ir_input.last_stats,
        gpu_peak_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)

    model.eval()
    with torch.no_grad():
        assisted, _ = model.independent_branch_prediction(
            'ir', **tensors, keep=keep, quality=quality)
        strict, _ = model.independent_branch_prediction(
            'ir', rgb=None, ir=tensors['ir'], depth=tensors['depth'],
            keep=keep, quality=quality)
    assert len(assisted) == len(strict)
    report['strict_and_geometry_assisted_paths'] = True

    save_mm_checkpoint(args.out / 'roundtrip.pt', model, epoch=0)
    restored, _ = load_mm_checkpoint(args.out / 'roundtrip.pt', device='cpu', weights=cfg.weights)
    assert restored.cfg.fusion.fusion_strategy == 'v521_stage_a_v1'
    report['checkpoint_roundtrip'] = True
    report['passed'] = True
    (args.out / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
