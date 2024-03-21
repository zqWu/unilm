#!/bin/bash

cd `dirname $0`
eval "$(conda shell.bash hook)"
conda activate wzq_dit

rm -rf ./wandb

#pkill wandb-service
#export http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890

#nohup python wandb_log.py >wandb.log 2>&1 &
python wandb_log.py
