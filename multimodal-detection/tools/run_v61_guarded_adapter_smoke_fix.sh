#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/DDBBPPT_v61/multimodal-detection
OUT=/root/v61_runs/v61_guarded_adapter_smoke_20260927_v2
PY=/root/miniconda3/envs/EFYOLO/bin/python
ROOT=/root/autodl-tmp/data/train_extracted
LABELS=/root/autodl-tmp/data/new_labels_2000
EXCLUDE=/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt
A0=/root/v61_runs/ir_a0_v61_a0_v11_full_s42_20260926_v1
INIT=/root/v61_runs/v61_quality_only_a12_b12_20260927_v1/stage_a/weights/best.pt

if [ -e "$OUT" ]; then
  echo "refusing existing output: $OUT" >&2
  exit 9
fi
cd "$REPO"
mkdir -p "$OUT"
printf 't0_running\n' > "$OUT/pipeline_state.txt"

"$PY" tools/test_v61_quality_input.py | tee "$OUT/t0_quality_adapter.log"
printf 'stage_a_running\n' > "$OUT/pipeline_state.txt"

"$PY" -u mm_yolo/train.py \
  --root "$ROOT" --labels "$LABELS" --exclude-stems "$EXCLUDE" \
  --out "$OUT" --name stage_a --weights /root/autodl-tmp/weights/yolo11s.pt \
  --modalities all --imgsz 736x1280 --epochs 2 --batch 16 --accum 1 --workers 12 \
  --architecture independent_p2_memory_v3 --precision bf16 --checkpoint-encoder \
  --memory-control bounded_v2 --depth-resampling nearest_valid_v2 \
  --ir-read-mode median_channel --ir-a0-cache "$A0" --require-ir-a0 \
  --metric-branch --sampler coverage --rare-extra-frac 0.1 --close-aug-frac 0 \
  --scale-min 1.0 --scale-max 1.0 --translate 0 --bn-policy adaptive_no_tail \
  --warmup 1 --lrf 0.5 --calibrate-clip-steps 32 --val-batch 8 \
  --lr 0.00002 --backbone-lr-mult 0.25 --fusion-lr-mult 1.0 \
  --p2-lr-mult 1.0 --detector-lr-mult 1.0 --semantic-lr-mult 1.0 \
  --ir-geometry-lr-mult 0.10 --weight-decay 0.0005 --nominal-batch 64 --grad-clip 1000 \
  --freeze-epochs 0 --train-stage aux_independent --aux-branch-mode both \
  --fusion-tier L2 --share-tier a --register-bus --depth-scales all \
  --depth-channels 4 --depth-view both --depth-init relative --no-prior --no-deformable \
  --no-dropout --misalign-px 0 --degrade-p 0 --rgb-color-p 0 --ir-noise-p 0 \
  --ir-gain-p 0 --depth-hole-p 0 --target-crop-p 0 --rotate-deg 0 \
  --target-occlusion-p 0 --ir-affine-p 0.50 --ir-affine-deg 5 \
  --ir-affine-shift 10 --ir-affine-scale 0.04 --rare-sample-max 2 \
  --val-ratio 0.2 --limit 320 --seed 42 --save-every 1 --val-every 1 \
  --val-limit 64 --val-conf 0.01 --eval-initial --init-checkpoint "$INIT" \
  --branch-aux-weight 0 --branch-aux-weights 0 1 0.6 \
  --branch-aux-end-weights 0 1 0.6 --flow-supervision-weight 0 \
  --flow-supervision-end-weight 0 --cross-modal-nce-weight 0 \
  --cross-modal-nce-end-weight 0 --p2-match-refine \
  --alignment-mode identity_residual_v2 --depth-reliability valid_support_v2 \
  --embedding-recon-weight 0 --embedding-recon-end-weight 0 \
  --embedding-alignment-weight 0 --embedding-alignment-end-weight 0 \
  --independent-preserve-weight 0 --independent-preserve-end-weight 0 \
  --fusion-strategy v521_stage_a_v1 --ir-affine-loss-weight 0.25 \
  --ir-affine-loss-end-weight 0.10 --evidence-supervision-weight 0 \
  --evidence-supervision-end-weight 0 \
  2>&1 | tee "$OUT/stage_a_console.log"

test -s "$OUT/stage_a/weights/best.pt"
printf 'stage_b_running\n' > "$OUT/pipeline_state.txt"

"$PY" -u mm_yolo/train.py \
  --root "$ROOT" --labels "$LABELS" --exclude-stems "$EXCLUDE" \
  --out "$OUT" --name stage_b --weights /root/autodl-tmp/weights/yolo11s.pt \
  --modalities all --imgsz 736x1280 --epochs 3 --batch 4 --accum 4 --workers 12 \
  --architecture independent_p2_memory_v3 --precision bf16 --checkpoint-encoder \
  --memory-control bounded_v2 --depth-resampling nearest_valid_v2 \
  --ir-read-mode median_channel --ir-a0-cache "$A0" --require-ir-a0 \
  --metric-branch --sampler coverage --rare-extra-frac 0.1 --close-aug-frac 0 \
  --scale-min 1.0 --scale-max 1.0 --translate 0 --bn-policy adaptive_no_tail \
  --warmup 1 --lrf 0.3 --calibrate-clip-steps 32 --val-batch 4 \
  --lr 0.0001 --rgb-stage-b-lr-mult 0 --backbone-lr-mult 0.1 \
  --fusion-lr-mult 1.0 --p2-lr-mult 0.1 --detector-lr-mult 0.1 \
  --semantic-lr-mult 0.1 --ir-geometry-lr-mult 0 \
  --weight-decay 0.0005 --nominal-batch 64 --grad-clip 1000 \
  --freeze-epochs 1 --train-stage residual_fusion --aux-branch-mode both \
  --fusion-tier L2 --share-tier a --register-bus --depth-scales all \
  --depth-channels 4 --depth-view both --depth-init relative --no-prior --no-deformable \
  --rgb-dropout 0.1 --aux-dropout 0.05 --dropout-start-epoch 1 \
  --misalign-px 0 --degrade-p 0 --rgb-color-p 0 --ir-noise-p 0 --ir-gain-p 0 \
  --depth-hole-p 0 --target-crop-p 0 --rotate-deg 0 --target-occlusion-p 0 \
  --ir-affine-p 0 --ir-affine-deg 5 --ir-affine-shift 10 --ir-affine-scale 0.04 \
  --rare-sample-max 2 --val-ratio 0.2 --limit 320 --seed 42 \
  --save-every 1 --val-every 1 --val-limit 64 --val-conf 0.01 --eval-initial \
  --init-checkpoint "$OUT/stage_a/weights/best.pt" \
  --branch-aux-weight 0 --branch-aux-weights 0 0 0 --branch-aux-end-weights 0 0 0 \
  --flow-supervision-weight 0 --flow-supervision-end-weight 0 \
  --cross-modal-nce-weight 0 --cross-modal-nce-end-weight 0 --p2-match-refine \
  --alignment-mode identity_residual_v2 --depth-reliability valid_support_v2 \
  --embedding-recon-weight 0 --embedding-recon-end-weight 0 \
  --embedding-alignment-weight 0 --embedding-alignment-end-weight 0 \
  --independent-preserve-weight 0.05 --independent-preserve-end-weight 0.05 \
  --fusion-strategy v521_stage_a_v1 --ir-affine-loss-weight 0 \
  --ir-affine-loss-end-weight 0 --evidence-supervision-weight 0 \
  --evidence-supervision-end-weight 0 \
  2>&1 | tee "$OUT/stage_b_console.log"

printf 'complete\n' > "$OUT/pipeline_state.txt"
