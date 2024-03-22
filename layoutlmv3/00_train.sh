#!/bin/bash

cd `dirname $0`
eval "$(conda shell.bash hook)"
conda activate wzq_layoutlmv3

_pwd=$(pwd)
export PYTHONPATH="$_pwd:$PYTHONPATH"

cd examples/object_detection
python train_net.py \
  --config-file cascade_layoutlmv3.yaml \
  --num-gpus 8 \
    MODEL.WEIGHTS $_pwd/models/layoutlmv3-base-chinese/pytorch_model.bin \
    OUTPUT_DIR $_pwd/models/output \
    PUBLAYNET_DATA_DIR_TRAIN $_pwd/data/publaynet/train \
    PUBLAYNET_DATA_DIR_TEST $_pwd/data/publaynet/val
