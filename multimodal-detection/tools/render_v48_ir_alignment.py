from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "mm_yolo"
for path in (str(ROOT), str(ROOT / "vendor"), str(MM)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import AugCfg, MMDataset, collate  # noqa: E402
from independent_fusion import affine_flow, warp  # noqa: E402
from model import load_mm_checkpoint, resolve_infer_canvas  # noqa: E402


def _u8(x: torch.Tensor) -> np.ndarray:
    return np.clip(x.detach().float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _edge_overlay(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edge = cv2.Canny(gray, 60, 140)
    ir_edge = cv2.Canny(ir, 45, 110)
    base = cv2.cvtColor((gray * 0.28).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    base[rgb_edge > 0] = (0, 255, 0)       # RGB: green
    base[ir_edge > 0] = (255, 0, 255)      # IR: magenta
    both = (rgb_edge > 0) & (ir_edge > 0)
    base[both] = (255, 255, 255)           # overlap: white
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, ck = load_mm_checkpoint(args.checkpoint, strict=True, device=device)
    model.eval()
    canvas = tuple(resolve_infer_canvas(model, (736, 1280)))
    filename = args.stem + ".png"
    sample = {
        "stem": args.stem,
        "files": {m: filename for m in ("visible", "infrared", "depth")},
        "boxes": np.zeros((0, 5), np.float32),
    }
    aug = AugCfg(imgsz=canvas, depth_resampling=model.cfg.depth_resampling,
                 legacy_lowlight=int(model.cfg.encoder.depth_input_channels) == 2)
    ds = MMDataset(Path(args.root), [sample], imgsz=canvas, train=False,
                   aug=aug, enabled=("rgb", "ir", "dep"))
    item = ds[0]
    if item is None:
        raise RuntimeError("failed to read requested sample")
    batch = collate([item])
    quality = {k: v.to(device) for k, v in batch["quality"].items()}
    keep = {k: v.to(device) for k, v in batch["keep"].items()}
    with torch.no_grad():
        model(batch["rgb"].to(device), batch["ir"].to(device),
              batch["depth"].to(device), quality=quality,
              prior=batch["prior"].to(device), keep=keep)
        raw = model._ir_affine_prediction[0].float()
        confidence = float(model._ir_affine_confidence[0, 0].float().cpu())
        max_angle = float(model.cfg.fusion.ir_affine_max_degrees)
        angle_deg = float(raw[0].cpu()) * max_angle
        params = raw.new_tensor([[np.deg2rad(angle_deg), 0.0, 0.0, 0.0]])
        source = batch["ir"].to(device)
        fully_warped = warp(source, affine_flow(params, source.shape[-2:]))
        used = source + confidence * (fully_warped - source)

    rgb = _u8(batch["rgb"][0]).transpose(1, 2, 0)
    ir = _u8(batch["ir"][0, 0])
    full = _u8(fully_warped[0, 0])
    aligned = _u8(used[0, 0])
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    info = (f"angle={angle_deg:+.4f} deg  confidence={confidence:.4f}  "
            f"small-angle effective~{angle_deg * confidence:+.4f} deg")
    panels = [
        _label(rgb_bgr, "RGB canvas"),
        _label(cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), "IR original canvas"),
        _label(cv2.cvtColor(full, cv2.COLOR_GRAY2BGR), f"IR full predicted rotation {angle_deg:+.3f} deg"),
        _label(cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR), "IR actually used: confidence blend"),
        _label(_edge_overlay(rgb, ir), "Before: RGB green / IR magenta / overlap white"),
        _label(_edge_overlay(rgb, aligned), "After: RGB green / IR magenta / overlap white"),
    ]
    thumb_w, thumb_h = 960, 552
    panels = [cv2.resize(p, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for p in panels]
    sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    cv2.rectangle(sheet, (0, sheet.shape[0] - 42), (sheet.shape[1], sheet.shape[0]), (0, 0, 0), -1)
    cv2.putText(sheet, info, (12, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{args.stem}_v48_ir_alignment_contact.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / f"{args.stem}_ir_original.png"), ir)
    cv2.imwrite(str(out / f"{args.stem}_ir_full_rotation.png"), full)
    cv2.imwrite(str(out / f"{args.stem}_ir_model_used.png"), aligned)
    metadata = {
        "stem": args.stem,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ck.get("epoch"),
        "checkpoint_best_map": ck.get("best_map"),
        "canvas": list(canvas),
        "raw_affine": [float(v) for v in raw.cpu()],
        "predicted_angle_degrees": angle_deg,
        "confidence": confidence,
        "small_angle_effective_degrees": angle_deg * confidence,
        "note": "Model applies feature-space confidence blending; pixel images reproduce the same rotation and blend for visualization.",
    }
    (out / f"{args.stem}_v48_ir_alignment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
