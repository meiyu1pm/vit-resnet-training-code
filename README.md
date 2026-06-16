# vit-resnet-training-code
This repo is update for learning vit construction
# 训练ViT与ResNet-18
### 1 环境配置
```shell
conda create -n vit python=3.10
pip install torch==2.x.0 torchvision==0.15.2
pip install -r requirements.txt
```
### 2 数据集准备
Cifar-10数据集：
./data/cifar-10-batches-py/
- 不单独上传data文件夹，请自行下载Cifar-10数据集并放置在./data/cifar-10-batches-py/下

### 3 启动训练
./scripts/train_resnet.sh
./scripts/train_vit.sh

### 4 模型参数配置
./configs/

### 5 模型保存
./train_results/

### 6 目录结构
```
├── configs
│   ├── __init__.py
│   ├── basic_config.py
│   ├── resnet_config.py
│   └── vit_config.py
├── data
│   └── cifar-10-batches-py
├── models
│   ├── __init__.py
│   ├── ResNet.py
│   └── ViT.py
├── scripts
│   ├── train_resnet.sh
│   └── train_vit.sh
├── train_results
├── utils
│   ├── train_utils.py
│   └── data_utils.py
├── train.py
└── readme.md
```
