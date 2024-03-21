# 系统
- ubuntu: 2004
- nvcc -v: 11.7
- python: 3.8
- nvidia-gpu: A800

# doclaynet dataset 
- https://github.com/DS4SD/DocLayNet
- 数据量 = 80863
	- train 69375
	- test   4999
	- val    6489

## labels
```
Caption
Footnote
Formula
List-item
Page-footer
Page-header
Picture
Section-header
Table
Text
Title
```

## download
```bash

echo `date +"%F %T"` >> log.txt
wget -O "DocLayNet_core.zip" "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
echo "DocLayNet_core.zip" >> log.txt
# 28G zip
```

## preprocess
```bash
unzip DocLayNet_core.zip # 
# COCO/test.json  train.json  val.json
# PNG/xxx.png


cd unilm/dit/object_detection
ln -s /data/datasets/doclaynet doclaynet_data #
```

# setup environment
```bash
conda create -n wzq_dit python=3.8
conda activate wzq_dit

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

cd unilm/dit
git clone -b v0.6 https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
pip install -r requirements.txt # requirements 有被修改
```

# setup doclaynet

## config
- 使用 cascade_dit_base
- 基本 copy pulaynet_configs, 以下注明修改的地方
- object_detection/doclaynet_configs/Base-RCNN-FPN.yaml
```yaml
MODEL:
  ROI_HEADS:
    NUM_CLASSES: 11
DATASETS:
  TRAIN: ("doclaynet_train",)
  TEST: ("doclaynet_val",)
SOLVER:
  BASE_LR: 0.0001
DATALOADER:
  NUM_WORKERS: 32
```
- object_detection/doclaynet_configs/cascade/cascade_dit_base.yaml

## train_net.py, 添加如下内容
```python
    register_coco_instances(
        "doclaynet_train", 
        {}, 
        "doclaynet_data/COCO/train.json", 
        "doclaynet_data/PNG")

    register_coco_instances(
        "doclaynet_val", 
        {}, 
        "doclaynet_data/COCO/val.json", 
        "doclaynet_data/PNG")
```

## object_detection/inference.py, 添加内容
```python
    elif cfg.DATASETS.TEST[0] == "doclaynet_val":
        md.set(thing_classes=[
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ])
```

## 添加 object_detection/01_train.sh

## 整体 object_detection 文件夹下的变动如下
```
.
├── 01_train.sh         # 新增, 训练脚本
├── doclaynet_configs   # 新增, doclaynet 配置
├── doclaynet_data      # 新增, doclaynet 数据
├── inference.py        # 修改, 增加 doclaynet配置
├── train_doclaynet.md  # 新增, 本文件
└── train_net.py        # 修改, 增加 doclaynet配置
```

# 测试
- `./01_train.sh`

