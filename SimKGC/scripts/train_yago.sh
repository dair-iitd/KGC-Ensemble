#!/usr/bin/env bash

set -x
set -e

TASK="YAGO3-10"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 64 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 2 \
--epochs 5 \
--workers 2 \
--max-to-keep 5 "$@"
