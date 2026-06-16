import os
from configs.basic_config import *

# 模型标识
MODEL_NAME = "resnet"

# ResNet-18 模型参数
IMG_SIZE = 32
IN_CHANNELS = 3
NUM_CLASSES = 10

# ResNet 适配的训练超参
MAX_LR = LR * 2
PCT_START = 0.2
ANNEAL_STRATEGY = "cos"
DIV_FACTOR = 20
FINAL_DIV_FACTOR = 100

# 专属保存路径（和 ViT 分开存放，不互相覆盖）
MODEL_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_model.pth")
METRICS_JSON_PATH = os.path.join(METRICS_DIR, f"{MODEL_NAME}_metrics.json")

# 补全通用训练参数（和basic_config对齐）
WEIGHT_DECAY = 0.03
LABEL_SMOOTHING = 0.1
RANDOM_SEED = 188
LR = 0.001
EPOCHS = 200
BATCH_SIZE = 64

# ResNet无需的参数（避免KeyError）
PATCH_SIZE = None
EMBED_DIM = None
DEPTH = None
NUM_HEADS = None
DROP_RATIO = None
ATTN_DROP_RATIO = None
DROP_PATH_RATIO = None
MLP_RATIO = None
