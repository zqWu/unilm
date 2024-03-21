#!/bin/bash

# select dataset 进行训练
config_file=doclaynet_configs/cascade/cascade_dit_base.yaml
pre_train_model=dit_base_patch16_224.pth
####################### doclaynet #######################
output_dir=output_$(date +%Y-%m-%d-%H-%M)

num_gpus=8
echo "config_file=$config_file"

read -n1 -p "run in foreground[y/n]: " foreground
echo ""
if [ -z "$foreground" ]; then foreground="y"; fi
foreground=$(echo "$foreground" | tr '[:upper:]' '[:lower:]')


if [ "$foreground" = "y" ]; then
    # 前台, 用于检测是否能跑
    echo "fore ground"
    python train_net.py --config-file $config_file --num-gpus $num_gpus MODEL.WEIGHTS $pre_train_model OUTPUT_DIR $output_dir
else
    # 后台
    echo "back ground"
    nohup python train_net.py --config-file $config_file --num-gpus $num_gpus MODEL.WEIGHTS $pre_train_model OUTPUT_DIR $output_dir >train.log 2>&1 &
    tail -f train.log
fi
