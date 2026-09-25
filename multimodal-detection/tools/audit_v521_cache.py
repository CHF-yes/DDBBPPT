"""Summarize the explicit V5.2.1 cache before model training."""
import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = {
    'ghost_mean': ('ghost_probability', 'mean'),
    'ghost_max': ('ghost_probability', 'max'),
    'ghost_conf': ('ghost_transform_confidence', 'scalar'),
    'thermal_mean': ('thermal_confidence_map', 'mean'),
    'hard_mean': ('hard_mask', 'mean'),
    'soft_mean': ('soft_mask', 'mean'),
    'exclude_mean': ('align_exclude_mask', 'mean'),
    'coarse_conf': ('thermal_transform_confidence', 'scalar'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    values = {key: [] for key in FIELDS}
    values.update(leak_proxy_mean=[], leak_proxy_max=[], coarse_usable=[])
    files = sorted((args.cache / 'samples').glob('*.npz'))
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            for key, (field, reducer) in FIELDS.items():
                x = np.asarray(z[field], np.float32)
                value = float(x.reshape(())) if reducer == 'scalar' else float(getattr(x, reducer)())
                values[key].append(value)
            q = np.asarray(z['quality_maps'], np.float32)
            values['leak_proxy_mean'].append(float(q[7].mean()))
            values['leak_proxy_max'].append(float(q[7].max()))
            values['coarse_usable'].append(float(np.asarray(
                z.get('v52_coarse_usable', 0), np.float32).reshape(())))
    summary = {}
    for key, rows in values.items():
        x = np.asarray(rows, np.float64)
        summary[key] = {
            'min': float(x.min()), 'p50': float(np.percentile(x, 50)),
            'p90': float(np.percentile(x, 90)), 'max': float(x.max()),
            'nonzero': int(np.count_nonzero(x > 1e-6)),
        }
    print(json.dumps({'samples': len(files), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
