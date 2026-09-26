# V4.8 minimum viable multimodal plan

Date: 2026-09-23

## Why V4.8 exists

V4.4 remains the protected baseline. Its validation best is 0.456662 mAP50-95.
V4.7 started from that checkpoint, peaked at 0.456812 on epoch 2, and fell to
0.4509 by epoch 18 while the auxiliary route ratios increased. This is evidence
that V4.7 learned more routing activity, but not more trustworthy information.

V4.8 therefore does not tune the V4.7 evidence gates. It preserves the complete
V4.4 detector function and adds a small, auditable plugin.

## Fixed experiment boundary

Unchanged in the first V4.8 run:

- YOLO11s, 736x1280 rectangular canvas, fixed seed-42 1600/400 split.
- Mosaic 0.04 and target occlusion 0.18.
- CIoU + DFL and standard per-image NMS.
- V4.4 best.pt is the only initialization checkpoint.

Changed in V4.8:

1. Disable IR-only artificial affine augmentation. Normal synchronized
   three-modality geometry remains enabled.
2. Estimate rotation-only RGB/IR correction from real paired features. Shift and
   scale are fixed to zero. The original IR grid remains the protected V4.4 base;
   corrected IR is used only by the additive V4.8 plugin.
3. Reuse the existing common/private spatial embeddings and cross-scale memory.
   Do not use the V4.7 chain `gain * gate * evidence * reliability`.
4. IR contributes an additive embedding complement. Depth does not receive an
   independent class rescue route; it provides a zero-initialized geometry
   support residual conditioned on RGB, IR and valid depth.
5. Train only the new plugin/alignment during the short frozen phase. Then release
   auxiliary embeddings/encoder plus P2, neck and detector at low role-specific
   learning rates so the single detector can adapt to the new information.

## Safety contracts

- Exact function preservation: immediately after migration and reset, V4.8 must
  produce the same detector tensors as V4.4.
- Original V4.4 fusion modules are frozen throughout V4.8.
- IR correction is rotation-only and confidence-blended; it never overwrites the
  original IR path.
- Depth support is masked by valid depth and writes only to localization geometry.
- Final inference still uses one learned fused detector, never detector voting.

## Preflight gates

Before a full run:

1. Unit tests for checkpoint migration, exact identity, non-zero plugin gradients,
   rotation-only flow, and depth-invalid no-op.
2. Dry-run command inspection: IR-only affine probability and affine supervision
   must be zero.
3. GPU smoke run: finite loss/gradients, no OOM, all 12 validation classes present.
4. Initial validation must match V4.4 within 0.0005 absolute mAP50-95.

## Training and decision rule

- Start from V4.4 stage_b/weights/best.pt.
- First 3 epochs: new alignment/complement plugin only.
- Remaining epochs: low-LR auxiliary embedding/encoder and P2/neck/detector
  adaptation; RGB backbone and original V4.4 fusion remain frozen.
- Keep validation every epoch and always deliver best.pt, not last.pt.
- Stop early as a failed direction if, after epoch 6, best mAP50-95 has not
  exceeded V4.4 by at least 0.002 and the plugin route has become non-trivial.

## Success criteria

- Minimum: stable +0.5 local mAP50-95 point over V4.4 without per-class collapse.
- IR ablation should reduce recall/AP on IR-visible targets.
- Depth ablation should mainly reduce high-IoU/localization metrics, not class AP.
- Long-object and occlusion subsets should improve at AP75/AP90 or complete-box
  recall; these subset checks are diagnostic and do not replace full validation.

