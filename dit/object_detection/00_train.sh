#!/bin/bash

# select dataset 进行训练
config_file=doclaynet_configs/cascade/cascade_dit_base.yaml
pre_train_model=dit_base_patch16_224.pth
####################### doclaynet #######################
output_dir=output_$(date +%Y-%m-%d-%H-%M)
num_gpus=8
echo "config_file=$config_file"


# background ?
read -n1 -p "run in foreground [y/n]: " foreground
echo ""
if [ -z "$foreground" ]; then foreground="y"; fi
foreground=$(echo "$foreground" | tr '[:upper:]' '[:lower:]')


# resume ?
read -p "resume dir (keep empty if not resume): " resume_dir
echo ""
if [ ! -z "$resume_dir" ]; then
    if [[ ! -d "$resume_dir" ]]; then
        echo "resume_dir $resume_dir not exists"
        exit 1
    fi
fi


if [ "$foreground" = "y" ]; then
    # 前台, 用于检测是否能跑
    echo "fore ground"
    if [ ! -z "$resume_dir" ]; then
        echo "resume $resume_dir"
        python train_net.py --resume --config-file $config_file --num-gpus $num_gpus OUTPUT_DIR $resume_dir
    else
        echo "new train"
        python train_net.py --config-file $config_file --num-gpus $num_gpus MODEL.WEIGHTS $pre_train_model OUTPUT_DIR $output_dir
    fi

else
    # 后台
    echo "back ground"
    if [ ! -z "$resume_dir" ]; then
        echo "resume $resume_dir"
        nohup python train_net.py --resume --config-file $config_file --num-gpus $num_gpus OUTPUT_DIR $resume_dir >>train.log 2>&1 &
        tail -f train.log
    else
        echo "new train"
        nohup python train_net.py --config-file $config_file --num-gpus $num_gpus MODEL.WEIGHTS $pre_train_model OUTPUT_DIR $output_dir >train.log 2>&1 &
        tail -f train.log
    fi
fi
