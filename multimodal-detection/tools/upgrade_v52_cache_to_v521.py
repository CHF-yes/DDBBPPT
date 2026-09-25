"""Create a versioned V5.2.1 cache without promoting weak candidates to labels."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def upgrade(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    sample_out = destination / 'samples'
    sample_out.mkdir()
    rows = []
    for path in sorted((source / 'samples').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            values = {key: z[key] for key in z.files}
        q = np.asarray(values['quality_maps'], np.float32)
        visible = np.asarray(values.get('visible_mask', np.ones_like(q[7])), np.float32)
        leak_conf = float(np.asarray(values.get('leakage_confidence', 0.)).reshape(()))
        # q[7] is the spatial RGB-leakage/ghost proxy.  Keep it even when the
        # global physical alpha fit is uncertain; confidence is a separate
        # input so uncertainty never erases the observation itself.
        ghost = np.clip(q[7], 0, 1)
        thermal = np.clip(q[8], 0, 1)
        hard = np.clip(1 - visible, 0, 1)
        soft = np.clip(.45 * q[5] + .35 * q[6] + .20 * q[4], 0, 1)
        exclude = hard.copy()
        by, bx = max(2, round(exclude.shape[-2] * .04)), max(2, round(exclude.shape[-1] * .04))
        exclude[:by] = exclude[-by:] = 1
        exclude[:, :bx] = exclude[:, -bx:] = 1
        values.update(
            v521_version=np.int32(1),
            ghost_probability=ghost.astype(np.float16),
            thermal_confidence_map=thermal.astype(np.float16),
            hard_mask=hard.astype(np.float16),
            soft_mask=soft.astype(np.float16),
            align_exclude_mask=exclude.astype(np.float16),
            thermal_transform_confidence=np.float32(values.get('affine_confidence', 0.)),
            ghost_transform_confidence=np.float32(leak_conf),
            coarse_candidate_available=np.uint8(values.get('v52_coarse_usable', 0)),
            coarse_candidate_only=np.uint8(not bool(values.get('affine_supervised', 0))),
        )
        target = sample_out / path.name
        np.savez_compressed(target, **values)
        rows.append({
            'stem': path.stem,
            'supervised': int(np.asarray(values['affine_supervised']).reshape(())),
            'candidate_only': int(np.asarray(values['coarse_candidate_only']).reshape(())),
            'thermal_confidence': float(np.asarray(values['thermal_transform_confidence']).reshape(())),
            'ghost_confidence': float(np.asarray(values['ghost_transform_confidence']).reshape(())),
        })
    for name in ('split_s42_v52.json', 'audit_summary.json'):
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
    manifest = {
        'version': 'v521_explicit_coarse_v1',
        'source': str(source.resolve()),
        'source_manifest_sha256': digest(source / 'manifest.json') if (source / 'manifest.json').is_file() else '',
        'samples': len(rows),
        'supervised': sum(row['supervised'] for row in rows),
        'candidate_only': sum(row['candidate_only'] for row in rows),
        'fields': ['ghost_probability', 'thermal_confidence_map', 'hard_mask',
                   'soft_mask', 'align_exclude_mask', 'thermal_transform_confidence',
                   'ghost_transform_confidence', 'coarse_candidate_available',
                   'coarse_candidate_only'],
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (destination / 'audit_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.destination), indent=2))


if __name__ == '__main__':
    main()
