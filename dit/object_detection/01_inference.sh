#!/bin/bash

cd `dirname $0`
eval "$(conda shell.bash hook)"
conda activate wzq_dit


############################
# 进行推理:
# 输入: image_input文件夹
# 输出: image_output_${use_model}_{日期}
############################

## 配置设定
gpu_id=1
config_file=doclaynet_configs/cascade/cascade_dit_base.yaml
model_path=models/model_0001999.pth
output_folder="image_output"

echo "model_path=$model_path"
echo "config_file=$config_file"
echo "output_folder=$output_folder"

read -p "press any key to run, ctrl+c to exit:"

inference_one_image() {
  input_file=$1
  output_file=$2

  CUDA_VISIBLE_DEVICES=$gpu_id \
  python ./inference.py \
  --image_path ${input_file} \
  --output_file_name ${output_file} \
  --config-file ${config_file} \
  --opts MODEL.WEIGHTS ${model_path}
}

# clear output
rm ${output_folder} -rf
mkdir ${output_folder}

# inference
ls image_input | while read one_image;do
  input_file="image_input/$one_image"

  if [ -f "$input_file" ]; then
    output_file="${output_folder}/${one_image}"
    echo "======> process $input_file => $output_file"
    inference_one_image $input_file $output_file
  else
    echo "skip $one_image"
  fi
done


# zip or not
read -n1 -p "need zip ? [n/y]: " need_zip
echo ""
if [ -z "$need_zip" ]; then need_zip="n"; fi
need_zip=$(echo "$need_zip" | tr '[:upper:]' '[:lower:]')

if [ "$need_zip" == "y" ]; then
  zip -r image_output.zip ${output_folder}
fi
