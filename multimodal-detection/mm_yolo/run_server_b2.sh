#!/usr/bin/env bash
# Linux/CUDA B2 launcher. Paths and capacity knobs come from environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to train_extracted}"
: "${LABEL_ROOT:?Set LABEL_ROOT to new_labels_2000}"
: "${WEIGHTS:?Set WEIGHTS to a local yolo11s.pt or yolo11m.pt}"

RUNS_DIR="${RUNS_DIR:-${SCRIPT_DIR}/../runs}"
RUN_NAME="${RUN_NAME:-b2_depth4_server}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-768x1376}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-2}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS:-100}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

args=(
  "${SCRIPT_DIR}/train.py"
  --root "${DATA_ROOT}"
  --labels "${LABEL_ROOT}"
  --weights "${WEIGHTS}"
  --out "${RUNS_DIR}"
  --name "${RUN_NAME}"
  --device "${DEVICE}"
  --modalities all
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --accum "${ACCUM}"
  --workers "${WORKERS}"
  --epochs "${EPOCHS}"
  --lr 0.0001
  --freeze-epochs 3
  --fusion-tier L2
  --share-tier c
  --register-bus
  --depth-scales p4p5
  --depth-channels 4
  --rgb-dropout 0.02
  --aux-dropout 0.02
  --dropout-start-epoch 10
  --misalign-px 5
  --degrade-p 0.15
  --ir-noise-p 0.10
  --depth-hole-p 0.10
  --target-crop-p 0.08
  --rare-sample-max 1.5
  --eval-initial
  --val-every 5
  --val-limit 0
  --val-conf 0.01
  --save-every 1
)

# Only initialize from a checkpoint built with the same YOLO size/structure.
# Leave empty for a fresh yolo11m B2 run; train.py will still use COCO WEIGHTS.
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

exec "${PYTHON_BIN}" "${args[@]}"
