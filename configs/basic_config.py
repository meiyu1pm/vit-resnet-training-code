import os
import platform
import torch
import random
import math
import numpy as np

# 路径配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT_DIR, 'data/cifar-10-batches-py')
SAVE_DIR = os.path.join(ROOT_DIR, 'train_results')
LOG_DIR = os.path.join(SAVE_DIR, 'logs')
METRICS_DIR = os.path.join(SAVE_DIR, 'metrics')
TEST_BATCH_PATH = os.path.join(DATA_ROOT, 'test_batch')

# 数据集划分
VAL_SIZE = 5000
TRAIN_SIZE = 50000 - VAL_SIZE

# 设备配置
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用MPS(Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device
DEVICE = get_device()

# 多进程加载
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4

# 随机种子
RANDOM_SEED = 188
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 训练通用超参
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 0.03
LABEL_SMOOTHING = 0.1

# CIFAR10 标准化
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# 计算总训练步数
def get_total_train_steps(train_size=TRAIN_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    batches_per_epoch = math.ceil(train_size / batch_size)
    return epochs * batches_per_epoch

def check_data_path():
    """检查CIFAR10数据路径是否存在"""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(
            f"\n❌ 数据路径不存在: {DATA_ROOT}\n"
            "请下载CIFAR10数据集并解压到该路径,下载地址:\n"
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
