import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'vendor', ROOT / 'mm_yolo'):
    sys.path.insert(0, str(path))

from data import AugCfg, MMDataset, build_index, collate, load_split
from model import load_mm_checkpoint


def quantiles(values):
    values = np.asarray(values, np.float64)
    return {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)}


@torch.no_grad()
def audit(checkpoint, cache, device):
    model, _ = load_mm_checkpoint(checkpoint, device=device,
                                  weights='/root/autodl-tmp/weights/yolo11s.pt')
    model.eval()
    root = Path('/root/autodl-tmp/data/train_extracted')
    index = build_index(root, Path('/root/autodl-tmp/data/new_labels_2000'),
                        exclude_stems=Path('/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt'))
    _, rows = load_split(cache / 'split_s42_v52.json', index)
    aug = AugCfg(
        imgsz=(736, 1280), scale_range=(1., 1.), translate=0., hflip_p=0.,
        ir_read_mode='median_channel', ir_a0_cache=str(cache), require_ir_a0=True,
        depth_resampling='nearest_valid_v2', misalign_px=0, mosaic_p=0,
        rgb_drop_p=0, aux_drop_p=0, degrade_p=0, rgb_color_p=0,
        ir_noise_p=0, ir_gain_p=0, target_crop_p=0, ir_affine_p=0,
        v521_explicit=True, v521_angle_deg=15., v521_shift_frac=.20,
        v521_scale_min=.80, v521_scale_max=1.25)
    dataset = MMDataset(root, rows, imgsz=(736, 1280), train=False, aug=aug,
                        enabled=['rgb', 'ir', 'dep'], seed=42)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                        collate_fn=collate, pin_memory=True)
    records = []
    for batch in loader:
        rgb = batch['rgb'].to(device, non_blocking=True)
        ir = batch['ir'].to(device, non_blocking=True)
        depth = batch['depth'].to(device, non_blocking=True)
        quality = {key: value.to(device, non_blocking=True)
                   for key, value in batch['quality'].items()}
        keep = {key: value.to(device, non_blocking=True)
                for key, value in batch['keep'].items()}
        model.independent_branch_prediction(
            'ir', rgb=rgb, ir=ir, depth=depth, keep=keep, quality=quality)
        confidence = model.v52_ir_input.last_global_logit.float().sigmoid().cpu().numpy()
        normalized = model.v52_ir_input.last_global_normalized.float().cpu().numpy()
        physical = model.v52_ir_input.last_global_physical.float().cpu().numpy()
        for stem, conf, norm, phys in zip(batch['stems'], confidence, normalized, physical):
            records.append({
                'stem': stem, 'confidence': float(conf),
                'normalized': [float(v) for v in norm],
                'angle_deg': float(phys[0]), 'tx_px': float(phys[1]),
                'ty_px': float(phys[2]), 'scale': float(phys[3]),
            })
    confidence = np.asarray([row['confidence'] for row in records])
    angle = np.asarray([row['angle_deg'] for row in records])
    tx = np.asarray([row['tx_px'] for row in records])
    ty = np.asarray([row['ty_px'] for row in records])
    scale = np.asarray([row['scale'] for row in records])
    normalized = np.asarray([row['normalized'] for row in records])
    translation = np.sqrt((tx * tx + ty * ty) / 2)
    accepted_translation = confidence * translation
    accepted_angle = confidence * np.abs(angle)
    report = {
        'checkpoint': str(checkpoint), 'samples': len(records),
        'confidence': quantiles(confidence),
        'confidence_over_0_5': int((confidence > .5).sum()),
        'confidence_over_0_9': int((confidence > .9).sum()),
        'angle_abs_deg': quantiles(np.abs(angle)),
        'translation_rms_px': quantiles(translation),
        'scale': quantiles(scale),
        'effective_angle_abs_deg': quantiles(accepted_angle),
        'effective_translation_rms_px': quantiles(accepted_translation),
        'boundary_saturated_samples': int((np.abs(normalized) > .95).any(1).sum()),
        'top_confidence': sorted(records, key=lambda row: row['confidence'], reverse=True)[:20],
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    device = torch.device('cuda:0')
    reports = [audit(path, args.cache, device) for path in args.checkpoint]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
