#!/usr/bin/env bash
set -euo pipefail

CODE=/root/autodl-tmp/DDBBPPT_v62/multimodal-detection
ROOT=/root/autodl-tmp/data/train_extracted
LABELS=/root/autodl-tmp/data/new_labels_2000
CACHE=/root/v61_runs/ir_a0_v61_a0_v11_full_s42_20260926_v1
INIT=/root/autodl-tmp/runs/v61_t0_real_batch_20260926_v3/v61_base_init.pt
SPLIT=$CODE/configs/split_s42.json
OUT=/root/v61_runs/v62_stage_a_v44_20260927_v1

if [[ -e "$OUT" ]]; then
  echo "REFUSE_EXISTING_OUTPUT=$OUT" >&2
  exit 9
fi
if [[ ! -d "$CACHE/samples" || ! -f "$INIT" ]]; then
  echo "MISSING_INPUT cache=$CACHE init=$INIT" >&2
  exit 9
fi
mkdir -p "$OUT"
cd "$CODE"
/root/miniconda3/envs/EFYOLO/bin/python -u mm_yolo/train.py \
  --root "$ROOT" --labels "$LABELS" \
  --exclude-stems /root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt \
  --out "$OUT" --name stage_a --weights /root/autodl-tmp/weights/yolo11s.pt \
  --device auto --modalities all --imgsz 736x1280 --epochs 36 --batch 16 --accum 1 \
  --architecture independent_p2_memory_v3 --precision bf16 --checkpoint-encoder \
  --memory-control bounded_v2 --depth-resampling nearest_valid_v2 \
  --ir-read-mode median_channel --ir-a0-cache "$CACHE" --require-ir-a0 \
  --metric-branch --sampler coverage --rare-extra-frac 0.1 --close-aug-frac 0.25 \
  --scale-min 0.90 --scale-max 1.10 --translate 0.03 --bn-policy adaptive_no_tail \
  --warmup 3 --lrf 0.10 --calibrate-clip-steps 96 --val-batch 8 --workers 12 \
  --lr 0.0002 --backbone-lr-mult 0.25 --fusion-lr-mult 1.0 --p2-lr-mult 1.0 \
  --detector-lr-mult 1.0 --semantic-lr-mult 1.0 --ir-geometry-lr-mult 0.10 \
  --weight-decay 0.0005 --nominal-batch 64 --grad-clip 1000 \
  --train-stage aux_independent --aux-branch-mode both --fusion-tier L2 --share-tier a \
  --register-bus --depth-scales all --depth-channels 4 --depth-view both \
  --depth-init relative --no-prior --no-deformable --no-dropout \
  --misalign-px 0 --degrade-p 0 --rgb-color-p 0 --ir-noise-p 0.10 --ir-gain-p 0.18 \
  --depth-hole-p 0.08 --target-crop-p 0.12 --rotate-deg 0 --target-occlusion-p 0 \
  --mosaic 0 --ir-affine-p 0.20 --ir-affine-deg 5 --ir-affine-shift 10 \
  --ir-affine-scale 0.04 --rare-sample-max 3 --prefetch --val-ratio 0.2 \
  --limit 0 --seed 42 --amp --save-every 5 --val-every 5 --eval-initial \
  --init-checkpoint "$INIT" --split-file "$SPLIT" --val-limit 0 --val-conf 0.001 \
  --branch-aux-weight 0 --flow-supervision-weight 0 --cross-modal-nce-weight 0 \
  --p2-match-refine --match-floor 0 --alignment-mode identity_residual_v2 \
  --depth-reliability valid_support_v2 --flow-identity-weight 0 \
  --branch-aux-weights 0 1 1 --branch-aux-end-weights 0 1 1 \
  --flow-supervision-end-weight 0 --cross-modal-nce-end-weight 0 \
  --embedding-recon-weight 0 --embedding-recon-end-weight 0 \
  --embedding-alignment-weight 0 --embedding-alignment-end-weight 0 \
  --independent-preserve-weight 0 --independent-preserve-end-weight 0 \
  --fusion-strategy v521_stage_a_v1 --ir-affine-loss-weight 0.50 \
  --ir-affine-loss-end-weight 0.35 --evidence-supervision-weight 0 \
  --evidence-supervision-end-weight 0 2>&1 | tee "$OUT/stage_a_console.log"
