#!/usr/bin/env bash
# Linux 单 GPU 启动模板。所有路径由环境变量传入，不包含本机 Windows 路径。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to the extracted training directory}"
: "${LABEL_ROOT:?Set LABEL_ROOT to new_labels_2000}"
: "${WEIGHTS:?Set WEIGHTS to a local yolo11s.pt or yolo11m.pt}"

RUNS_DIR="${RUNS_DIR:-${SCRIPT_DIR}/../runs}"
RUN_NAME="${RUN_NAME:-b1_register_s}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
  --root "${DATA_ROOT}" \
  --labels "${LABEL_ROOT}" \
  --weights "${WEIGHTS}" \
  --out "${RUNS_DIR}" \
  --name "${RUN_NAME}" \
  --device "${DEVICE}" \
  --modalities all \
  --imgsz 544x960 \
  --batch "${BATCH}" \
  --accum "${ACCUM}" \
  --workers "${WORKERS}" \
  --epochs "${EPOCHS}" \
  --freeze-epochs 5 \
  --fusion-tier L2 \
  --share-tier c \
  --register-bus \
  --depth-scales p4p5 \
  --depth-channels 2 \
  --rgb-dropout 0.05 \
  --aux-dropout 0.05 \
  --dropout-start-epoch 5 \
  --misalign-px 15 \
  --degrade-p 0.30 \
  --ir-noise-p 0 \
  --depth-hole-p 0 \
  --target-crop-p 0 \
  --rare-sample-max 1 \
  --val-every 5 \
  --val-limit 0 \
  --save-every 1
