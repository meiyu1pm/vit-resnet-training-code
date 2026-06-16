import os
# 仅导入必要的工具/路径，而非全量导入*，避免参数混乱
from configs.basic_config import SAVE_DIR, METRICS_DIR, get_device

# ====================== 【ViT模型专属参数】（仅改这里即可调整模型结构） ======================
MODEL_NAME = "vit"
IMG_SIZE = 32          # 输入图片尺寸
PATCH_SIZE = 4         # 补丁尺寸
IN_CHANNELS = 3        # 输入通道数
NUM_CLASSES = 10       # 分类数（CIFAR10=10）
EMBED_DIM = 128        # 嵌入维度
DEPTH = 6              # Transformer块数
NUM_HEADS = 4          # 注意力头数
DROP_RATIO = 0.1       # Dropout比例
ATTN_DROP_RATIO = 0.1  # 注意力层Dropout比例
DROP_PATH_RATIO = 0.1  # DropPath比例
MLP_RATIO = 4          # MLP隐藏层维度放大倍数
QKV_BIAS = True        # QKV是否加偏置

# ====================== 【ViT训练专属参数】（仅改这里即可调整训练策略） ======================
# 基础训练超参（覆盖basic_config的默认值，无需改basic_config）
RANDOM_SEED = 188      # 随机种子
BATCH_SIZE = 64        # 批次大小
EPOCHS = 200           # 训练轮数
LR = 0.0005            # 初始学习率
WEIGHT_DECAY = 0.03    # 权重衰减
LABEL_SMOOTHING = 0.1  # 标签平滑系数

# 学习率调度器参数（OneCycleLR）
MAX_LR = LR * 4        # 最大学习率
PCT_START = 0.3        # 学习率上升阶段占比
ANNEAL_STRATEGY = "cos"# 退火策略（cos/linear）
DIV_FACTOR = 25        # 初始学习率 = MAX_LR / div_factor
FINAL_DIV_FACTOR = 100 # 最终学习率 = 初始LR / final_div_factor

# ====================== 【路径配置】（自动关联，无需手动改） ======================
# ViT专属模型保存路径
MODEL_PATH = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_model.pth")
# ViT专属指标保存路径
METRICS_JSON_PATH = os.path.join(METRICS_DIR, f"{MODEL_NAME}_metrics.json")

# ====================== 【复用工具函数】（和basic_config逻辑一致，无需改） ======================
# 设备配置（复用basic_config的逻辑）
DEVICE = get_device()
# 数据集划分（如需调整，直接在这里改，无需动basic_config）
VAL_SIZE = 5000
TRAIN_SIZE = 50000 - VAL_SIZE
# 多进程加载（如需调整，直接在这里改）
import platform
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
# CIFAR10标准化（如需调整，直接在这里改）
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
