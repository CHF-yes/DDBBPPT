#!/usr/bin/env bash
set -euo pipefail

CODE=/root/autodl-tmp/DDBBPPT_v61/multimodal-detection
OLD_CACHE=/root/autodl-tmp/cache/ir_a0_v61_a0_v6_256_s42_20260926_v1
CACHE=/root/v61_runs/ir_a0_v61_a0_v11_256_s42_20260926_v1
OUT=/root/v61_runs/v61_short_stage_a_a0_v11_256x2_20260926_v1

mkdir -p "$OUT"
if [[ -e "$CACHE" ]]; then
  echo "REFUSE_EXISTING_CACHE=$CACHE" >&2
  exit 9
fi
if [[ -e "$OUT/a1_adapter_frozen" || -e "$OUT/a2_adapter_trainable" ]]; then
  echo "REFUSE_EXISTING_TRAINING=$OUT" >&2
  exit 9
fi

cd "$CODE"
/root/miniconda3/envs/EFYOLO/bin/python -u tools/build_ir_a0_cache.py \
  --root /root/autodl-tmp/data/train_extracted \
  --labels /root/autodl-tmp/data/new_labels_2000 \
  --exclude-stems /root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt \
  --out "$CACHE" --old-cache "$OLD_CACHE" \
  --limit 256 --work-width 480 --preview-count 12 \
  --min-confidence 0.45 --workers 4 --opencv-threads 4 \
  --contract-canvas 736x1280 --contract-angle 25 \
  --contract-shift 16 --contract-scale 0.04 \
  2>&1 | tee "$OUT/cache_build.log"

cp "$OLD_CACHE/split_s42_v52.json" "$CACHE/split_s42_v52.json"
find "$OLD_CACHE/samples" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/old_sample_names.txt"
find "$CACHE/samples" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/new_sample_names.txt"
diff -u "$OUT/old_sample_names.txt" "$OUT/new_sample_names.txt" > "$OUT/sample_names.diff"
sha256sum "$OLD_CACHE/split_s42_v52.json" "$CACHE/split_s42_v52.json" > "$OUT/split_sha256.txt"

bash tools/launch_v61_short_stage_a.sh pair 2>&1 | tee "$OUT/stage_a_console.log"
