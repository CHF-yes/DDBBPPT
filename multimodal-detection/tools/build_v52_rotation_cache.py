"""Freeze historical J.C3 angle proposals; no GT-based acceptance or affine labels."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mm_yolo'))
from ir_a0 import save_sample
from data import quality_maps, read_ir_bundle


def conservative_border_masks(ir):
    """Infer only near-constant exterior fill as a hard invalid mask.

    This is intentionally self-contained so the production V5.2 cache builder
    does not depend on any historical alignment probe.
    """
    gray = np.median(ir, axis=2) if ir.ndim == 3 else ir
    high = np.max(ir, axis=2) if ir.ndim == 3 else ir
    low = np.min(ir, axis=2) if ir.ndim == 3 else ir
    h, w = gray.shape
    x = gray.astype(np.float64)
    var = np.maximum(0.0, cv2.blur(x * x, (7, 7)) - cv2.blur(x, (7, 7)) ** 2)
    near = ((high <= 8) & (high - low <= 3)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near, 8)
    hard = np.zeros_like(near)
    uncertain = np.zeros_like(near)
    components = []
    for index in range(1, n):
        xx, yy, ww, hh, area = stats[index]
        sides = [xx == 0, yy == 0, xx + ww >= w, yy + hh >= h]
        if not any(sides) or area < max(30, 0.0005 * h * w):
            continue
        region = (labels == index).astype(np.uint8)
        uncertain |= region
        frontier = cv2.dilate(region, np.ones((3, 3), np.uint8)) - region
        frontier[:2] = 0
        frontier[-2:] = 0
        frontier[:, :2] = 0
        frontier[:, -2:] = 0
        coords = np.column_stack(np.where(frontier))
        if len(coords) < 20:
            continue
        lines = cv2.HoughLinesP(
            frontier, 1, np.pi / 360, threshold=20,
            minLineLength=max(20, 0.10 * min(h, w)), maxLineGap=6)
        segments = np.asarray(lines).reshape(-1, 4) if lines is not None else []
        longest = max(
            (np.hypot(v[2] - v[0], v[3] - v[1]) for v in segments),
            default=0.0)
        samples = gray[region > 0]
        constant = np.percentile(samples, 95) - np.percentile(samples, 5) <= 4
        contrast = np.median(gray[frontier > 0]) - np.median(samples)
        shape_ok = (
            sum(sides) >= 2
            or ((sides[0] or sides[2]) and hh >= 0.35 * h and ww <= 0.3 * w)
            or ((sides[1] or sides[3]) and ww >= 0.35 * w and hh <= 0.3 * h))
        accepted = bool(
            constant and contrast >= 5 and shape_ok and area <= 0.40 * h * w
            and longest >= 0.20 * len(coords))
        if accepted:
            hard |= region
        components.append({
            "area": int(area),
            "hard": accepted,
            "frontier_contrast": float(contrast),
            "longest_interface": float(longest),
            "connected_sides": int(sum(sides)),
        })
    soft = np.clip(0.65 * uncertain + 0.35 * (np.sqrt(var) < 2), 0, 1).astype(np.float32)
    border = max(1, round(0.06 * min(h, w)))
    outer = np.zeros_like(near)
    outer[:border] = 1
    outer[-border:] = 1
    outer[:, :border] = 1
    outer[:, -border:] = 1
    distance = cv2.distanceTransform(1 - uncertain, cv2.DIST_L2, 5)
    exclude = (outer > 0) | (distance <= 10)
    return {
        "hard": hard,
        "soft": soft,
        "align": (~exclude).astype(np.uint8),
        "exclude": exclude.astype(np.uint8),
        "availability": np.ones_like(near),
        "components": components,
    }


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def process(job):
    row, root, out, reject = job
    cv2.setNumThreads(1)
    stem = row['stem']
    paths = {}
    for mod in ('visible', 'infrared', 'depth'):
        found = list((Path(root) / mod).glob(stem + '.*'))
        if len(found) != 1:
            raise ValueError((stem, mod, 'missing or ambiguous source'))
        paths[mod] = found[0]
    ir = cv2.imread(str(paths['infrared']), cv2.IMREAD_UNCHANGED)
    if ir is None:
        raise ValueError(stem)
    raw_dtype = ir.dtype
    if ir.ndim == 2:
        ir = np.repeat(ir[..., None], 3, 2)
    ir = ir[..., :3].astype(np.float32) / (256 if raw_dtype == np.uint16 else 1)
    shape = ir.shape[:2]
    if list(shape) != row['shape']:
        raise ValueError('geometry source shape changed: ' + stem)
    # Current conservative hard/soft masks; do not reuse the older mask that
    # treated bright/dark boundary buildings as outside the sensor FOV.
    f = min(1., 640 / shape[1])
    small = cv2.resize(ir, (round(shape[1]*f), round(shape[0]*f)))
    m = conservative_border_masks(small)
    gray, chroma = read_ir_bundle(paths['infrared'], mode='median_channel')
    gray = cv2.resize(gray, small.shape[1::-1], interpolation=cv2.INTER_AREA)
    chroma = cv2.resize(chroma, small.shape[1::-1], interpolation=cv2.INTER_AREA)
    own = quality_maps(None, gray, None, None, (96, 160), ir_chroma=chroma)['ir']
    q = np.zeros((10, 96, 160), np.float32)
    q[:8] = own[:8]
    q[3] = cv2.resize(m['hard'].astype(np.float32), (160, 96), interpolation=cv2.INTER_AREA)
    q[8] = (1-q[3]) * own[1] * (1-.4*own[5])
    t = row['thermal_transform']
    # Only image-based search diagnostics are read. Ignore object_gain,
    # object_coverage, thermal_supervised and all GT-derived approval fields.
    angle = float(t['angle'])
    usable = (stem not in reject and np.isfinite(angle) and abs(angle) < 24.75
              and float(t.get('multires_angle_delta', 99)) <= 2
              and min(float(t['score']), float(t['multires']['score'])) > .12
              and float(t['overlap']) > .5)
    adopted = angle if usable else 0.
    matrix = cv2.getRotationMatrix2D(((shape[1]-1)/2, (shape[0]-1)/2), adopted, 1.)
    # Coarse candidate is an input option, never a ground-truth rotation label.
    # affine_supervised=0 also means tx/ty/scale are NOT zero-motion labels.
    save_sample(Path(out)/'samples'/f'{stem}.npz', stem=stem,
                params=[0, 0, 0, 1], confidence=0., quality=q,
                visible=1-m['hard'], geometry=1-m['hard'], meta={}, shape=shape,
                sequence=row['sequence'], sequence_prior=[0, 0, 0, 1],
                sequence_confidence=0., affine_supervised=False)
    p = Path(out)/'samples'/f'{stem}.npz'
    with np.load(p) as z:
        arrays = {k:z[k] for k in z.files}
    arrays.update(v52_coarse_matrix=matrix.astype(np.float32),
                  v52_angle_ccw=np.float32(adopted), v52_candidate_angle=np.float32(angle),
                  v52_coarse_usable=np.uint8(usable),
                  v52_source_sha256=np.asarray(sha(paths['infrared'])))
    np.savez_compressed(p, **arrays)
    return dict(stem=stem, candidate=angle, adopted=adopted, usable=bool(usable),
                hard_fraction=float(m['hard'].mean()), cache_sha256=sha(p))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','audit','split','exclude','reject','out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    readlist=lambda path:{s.strip() for s in Path(path).read_text().splitlines()
                          if s.strip() and not s.lstrip().startswith('#')}
    exclude, reject = readlist(a.exclude), readlist(a.reject)
    split=json.loads(Path(a.split).read_text())
    split['train']=[s for s in split['train'] if s not in exclude]
    split['val']=[s for s in split['val'] if s not in exclude]
    (out/'split_s42_v52.json').write_text(json.dumps(split,indent=2))
    rows=[r for r in json.loads(Path(a.audit).read_text())['samples'] if r['stem'] not in exclude]
    assert set(split['train']+split['val']) == {r['stem'] for r in rows}
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(process,[(r,a.root,a.out,reject) for r in rows],chunksize=1):
            results.append(r)
            if len(results)%100==0: print(f'[V5.2 cache] {len(results)}/{len(rows)}',flush=True)
    summary=dict(version='v52_rotation_input_v1', n_samples=len(results),
                 train=len(split['train']),val=len(split['val']),exclude=sorted(exclude),
                 audit_sha256=sha(a.audit),script_sha256=sha(__file__),
                 angle_quantiles=np.quantile([r['adopted'] for r in results],[0,.1,.5,.9,1]).tolist(),
                 usable=sum(r['usable'] for r in results),
                 affine_supervised=0,translation_labels=0,scale_labels=0,
                 selection_uses_gt=False,stage_b_allowed=False,samples=results)
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='samples'}),flush=True)


if __name__=='__main__':
    main()
