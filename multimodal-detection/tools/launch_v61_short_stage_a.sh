#!/usr/bin/env bash
set -euo pipefail

CODE=/root/autodl-tmp/DDBBPPT_v61/multimodal-detection
OUT=/root/autodl-tmp/runs/v61_short_validation_20260926_v1
INIT=/root/autodl-tmp/runs/v61_t0_real_batch_20260926_v1/v61_base_init.pt
CACHE=/root/autodl-tmp/cache/ir_a0_v521_explicit_v2

mkdir -p "$OUT"
echo '4ee9ad69f8dcbc3296b63411ad5c8fef5af06f76e9c867d9a25f84bb72067467  /root/autodl-tmp/runs/v61_t0_real_batch_20260926_v1/v61_base_init.pt' | sha256sum -c -
git -C /root/autodl-tmp/DDBBPPT_v61 rev-parse HEAD > "$OUT/code_commit.txt"

run_stage_a() {
  local name="$1"
  shift
  cd "$CODE"
  /root/miniconda3/envs/EFYOLO/bin/python -u mm_yolo/train.py \
    --root /root/autodl-tmp/data/train_extracted \
    --labels /root/autodl-tmp/data/new_labels_2000 \
    --exclude-stems /root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt \
    --out "$OUT" --name "$name" \
    --weights /root/autodl-tmp/weights/yolo11s.pt \
    --device auto --modalities all --imgsz 736x1280 \
    --epochs 2 --batch 4 --accum 2 --architecture independent_p2_memory_v3 \
    --precision bf16 --checkpoint-encoder --mosaic 0 \
    --memory-control bounded_v2 --depth-resampling nearest_valid_v2 \
    --ir-read-mode median_channel --ir-a0-cache "$CACHE" \
    --require-ir-a0 --metric-branch --sampler coverage \
    --rare-extra-frac 0.1 --close-aug-frac 0.25 \
    --scale-min 1.0 --scale-max 1.0 --translate 0 \
    --bn-policy adaptive_no_tail --warmup 1 --lrf 0.2 \
    --calibrate-clip-steps 32 --val-batch 2 \
    --workers 4 --lr 0.0001 --backbone-lr-mult 0.25 \
    --ir-geometry-lr-mult 0.1 \
    --weight-decay 0.0005 --nominal-batch 64 --grad-clip 60 \
    --freeze-epochs 0 --train-stage aux_independent \
    --aux-branch-mode both --fusion-tier L2 --share-tier a \
    --register-bus --depth-scales all --depth-channels 4 \
    --depth-view both --depth-init relative --no-prior --no-deformable \
    --rgb-dropout 0 --aux-dropout 0 --dropout-start-epoch 0 --no-dropout \
    --misalign-px 0 --degrade-p 0 --rgb-color-p 0 \
    --ir-noise-p 0 --ir-gain-p 0 --depth-hole-p 0 \
    --target-crop-p 0 --rotate-deg 0 --target-occlusion-p 0 \
    --ir-affine-p 0.20 --ir-affine-deg 5 --ir-affine-shift 10 \
    --ir-affine-scale 0.04 --rare-sample-max 3 --prefetch \
    --val-ratio 0.2 --seed 42 --amp --save-every 1 \
    --val-every 1 --eval-initial --init-checkpoint "$INIT" \
    --split-file "$CACHE/split_s42_v52.json" \
    --limit 256 --val-limit 64 --val-conf 0.001 --branch-aux-weight 0 \
    --flow-supervision-weight 0 --cross-modal-nce-weight 0 \
    --p2-match-refine --match-floor 0 \
    --alignment-mode identity_residual_v2 \
    --depth-reliability valid_support_v2 --flow-identity-weight 0 \
    --branch-aux-weights 1 1 0.6 \
    --branch-aux-end-weights 1 1 0.6 \
    --flow-supervision-end-weight 0 --cross-modal-nce-end-weight 0 \
    --embedding-recon-weight 0 --embedding-recon-end-weight 0 \
    --embedding-alignment-weight 0 --embedding-alignment-end-weight 0 \
    --independent-preserve-weight 0 --independent-preserve-end-weight 0 \
    --fusion-strategy v521_stage_a_v1 \
    --ir-affine-loss-weight 0.50 --ir-affine-loss-end-weight 0.35 \
    --evidence-supervision-weight 0 --evidence-supervision-end-weight 0 \
    "$@"
}

run_stage_a a1_adapter_frozen --freeze-v52-ir-input-stage-a
run_stage_a a2_adapter_trainable

echo completed > "$OUT/stage_a_pair_status.txt"
