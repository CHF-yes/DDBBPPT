"""Read-only dataset/label audit; writes only a separate report, never labels/split."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))
from data import build_index, load_split, CoverageRareSampler, val_class_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    index = build_index(Path(args.root), Path(args.labels))
    train, val = load_split(Path(args.split), index)
    assert len(train) == 1600 and len(val) == 400
    assert not ({s['stem'] for s in train} & {s['stem'] for s in val})
    sampler = CoverageRareSampler(train, seed=42, extra_frac=.1)
    order = list(sampler)
    count = Counter(i for i, _, _ in order)
    issues = {"malformed": [], "out_of_image": [], "duplicate_rows": [], "missing_modality": []}
    for sample in index:
        source = Path(args.labels) / (sample["stem"] + ".txt")
        rows = []
        for line, row in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not row.strip():
                continue
            try:
                v = np.array([float(x) for x in row.split()])
                assert len(v) == 5 and np.isfinite(v).all()
                assert int(v[0]) == v[0] and 0 <= v[0] < 12 and (v[3:] > 0).all()
            except (ValueError, AssertionError):
                issues["malformed"].append([sample["stem"], line, row])
                continue
            if min(v[1:3] - v[3:5]/2) < -1e-5 or max(v[1:3] + v[3:5]/2) > 1+1e-5:
                issues["out_of_image"].append([sample["stem"], line])
            key = tuple(v)
            if key in rows:
                issues["duplicate_rows"].append([sample["stem"], line])
            rows.append(key)
        for modality in ("visible", "infrared", "depth"):
            if modality not in sample["files"] or not (Path(args.root)/modality/sample["files"][modality]).is_file():
                issues["missing_modality"].append([sample["stem"], modality])
    report = {"train_images": len(train), "val_images": len(val),
              "train_boxes": val_class_stats(train), "val_boxes": val_class_stats(val),
              "epoch_draws": len(order), "unique_images": len(count), "max_draws_per_image": max(count.values()),
              "epoch_boxes_including_extra": val_class_stats([train[i] for i, _, _ in order]),
              "issues": issues,
              "interpretation": "Boundary flags/duplicates are reported, not auto-edited. Class/annotation correctness requires visual review."}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**{k:v for k,v in report.items() if k != 'issues'},
                      "issue_counts": {k:len(v) for k,v in issues.items()}}, ensure_ascii=False, indent=2))
    if issues["malformed"] or issues["missing_modality"]:
        raise RuntimeError("Blocking dataset issues; inspect report before training")


if __name__ == "__main__":
    main()
