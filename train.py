import os
import json
import argparse
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import OneCycleLR
# 新增：导入动态配置函数
from configs import get_model_config
from configs.basic_config import set_seed, get_device, get_total_train_steps, check_data_path, TRAIN_SIZE, VAL_SIZE, DATA_ROOT, NUM_WORKERS
# 移除：不再直接导入vit_config的硬编码参数
# from configs.vit_config import *
from utils.data_utils import build_train_val_loader, build_test_loader
from utils.train_utils import evaluate, load_checkpoint
from models import build_model

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10分类训练(ViT/ResNet)")
    # 基础参数
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "resnet"], help="模型类型")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次（默认用配置文件值）")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小（默认用配置文件值）")
    parser.add_argument("--lr", type=float, default=None, help="初始学习率（默认用配置文件值）")
    parser.add_argument("--device", type=str, default=None, help="设备(cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认用配置文件值）")
    # 路径参数
    parser.add_argument("--model-path", type=str, default=None, help="模型断点路径（默认用配置文件值）")
    parser.add_argument("--metrics-path", type=str, default=None, help="指标保存路径（默认用配置文件值）")
    # 训练控制
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--only-test", action="store_true", help="仅测试")
    return parser.parse_args()

class Trainer:
    def __init__(self, args, model_cfg, model, criterion, optimizer, scheduler, metrics):
        # 新增：传入模型专属配置
        self.model_cfg = model_cfg
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        # 设备
        self.device = torch.device(args.device) if args.device else get_device()
        self.model.to(self.device)

        # 路径初始化（优先用命令行参数，否则用配置文件）
        self.model_path = args.model_path if args.model_path else model_cfg["MODEL_PATH"]
        self.metrics_path = args.metrics_path if args.metrics_path else model_cfg["METRICS_JSON_PATH"]
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

        # 训练状态
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.cur_epoch = 0
        # 优先用命令行epochs，否则用配置（basic_config）
        self.epochs = args.epochs if args.epochs else model_cfg.get("EPOCHS", 200)

        # 数据加载器
        self.train_dataloader = None
        self.val_dataloader = None

    def load_data(self):
        try:
            check_data_path()
        except FileNotFoundError as e:
            print(e)
            exit(1)
        # 构造数据路径
        train_batch_list = [os.path.join(DATA_ROOT, f'data_batch_{i}') for i in range(1, 6)]
        # 修复：使用模型专属的IMG_SIZE，且传入NUM_WORKERS（解决Windows兼容）
        self.train_dataloader, self.val_dataloader = build_train_val_loader(
            train_batch_list=train_batch_list,
            val_size=VAL_SIZE,
            train_size=TRAIN_SIZE,
            batch_size=self.args.batch_size if self.args.batch_size else self.model_cfg["BATCH_SIZE"],
            img_size=self.model_cfg["IMG_SIZE"],
            num_workers=NUM_WORKERS  # 新增：传递多进程参数
        )

    def load_checkpoint(self) -> bool:
        if not self.args.resume or not os.path.exists(self.model_path):
            print("ℹ️ 不续训/断点不存在，从头训练")
            return False
        # 加载断点（增加异常捕获）
        try:
            self.start_epoch, self.best_val_acc = load_checkpoint(
                self.model, self.optimizer, self.model_path, self.device
            )
            print(f"✅ 续训成功：起始轮次{self.start_epoch}，最优验证精度{self.best_val_acc:.2f}%")
        except Exception as e:
            print(f"❌ 加载断点失败: {e}，将从头训练")
            return False
        # 加载指标（增加异常捕获）
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                print(f"✅ 加载历史指标成功")
            except Exception as e:
                print(f"⚠️ 加载指标失败: {e}，使用空指标")
        return True

    # 其余方法（_save_metrics/save_best_model/test）保持不变，仅修改test方法中的img_size：
    @torch.no_grad()
    def test(self):
        print("\n" + "="*50)
        print("===== 测试集最终评估 =====")
        if os.path.exists(self.model_path):
            try:
                load_checkpoint(self.model, self.optimizer, self.model_path, self.device)
            except Exception as e:
                print(f"❌ 加载测试模型失败: {e}")
                return None, None
        test_loader = build_test_loader(
            test_batch_path=os.path.join(DATA_ROOT, 'test_batch'),
            batch_size=self.args.batch_size if self.args.batch_size else self.model_cfg["BATCH_SIZE"],
            img_size=self.model_cfg["IMG_SIZE"],  # 改用模型专属IMG_SIZE
            num_workers=NUM_WORKERS  # 新增：Windows兼容
        )
        test_loss, test_acc = evaluate(self.model, test_loader, self.criterion, self.device)
        print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")
        self.metrics["test_loss"] = float(test_loss)
        self.metrics["test_acc"] = float(test_acc)
        self._save_metrics()
        return test_loss, test_acc

    # train方法无需修改，仅循环的epochs已从model_cfg获取

if __name__ == '__main__':
    # 1. 解析参数
    args = parse_args()
    # 2. 动态加载模型配置（核心修复）
    model_cfg = get_model_config(args.model)
    # 3. 设置种子（优先命令行，否则用配置）
    seed = args.seed if args.seed else model_cfg.get("RANDOM_SEED", 188)
    set_seed(seed)
    # 4. 构建模型（使用动态配置，避免硬编码）
    model_kwargs = {
        "model_name": args.model,
        "img_size": model_cfg["IMG_SIZE"],
        "in_c": model_cfg["IN_CHANNELS"],
        "num_classes": model_cfg["NUM_CLASSES"],
        "weight_decay": model_cfg["WEIGHT_DECAY"]
    }
    # ViT专属参数（动态判断）
    if args.model == "vit":
        model_kwargs.update({
            "patch_size": model_cfg["PATCH_SIZE"],
            "embed_dim": model_cfg["EMBED_DIM"],
            "depth": model_cfg["DEPTH"],
            "num_heads": model_cfg["NUM_HEADS"],
            "drop_ratio": model_cfg["DROP_RATIO"],
            "attn_drop_ratio": model_cfg["ATTN_DROP_RATIO"],
            "drop_path_ratio": model_cfg["DROP_PATH_RATIO"],
            "mlp_ratio": model_cfg["MLP_RATIO"]
        })
    # 构建模型
    model = build_model(**model_kwargs)

    # 5. 损失/优化器/调度器（使用动态配置）
    criterion = nn.CrossEntropyLoss(label_smoothing=model_cfg.get("LABEL_SMOOTHING", 0.1))
    # 学习率（优先命令行，否则用配置）
    lr = args.lr if args.lr else model_cfg["LR"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model_cfg["WEIGHT_DECAY"])
    # 批次大小（优先命令行，否则用配置）
    batch_size = args.batch_size if args.batch_size else model_cfg["BATCH_SIZE"]
    # 总步数计算
    total_steps = get_total_train_steps(TRAIN_SIZE, batch_size, args.epochs if args.epochs else model_cfg["EPOCHS"])
    # 调度器参数从模型配置获取
    scheduler = OneCycleLR(
        optimizer=optimizer, 
        max_lr=model_cfg["MAX_LR"], 
        total_steps=total_steps,
        pct_start=model_cfg["PCT_START"], 
        anneal_strategy=model_cfg["ANNEAL_STRATEGY"],
        div_factor=model_cfg["DIV_FACTOR"], 
        final_div_factor=model_cfg["FINAL_DIV_FACTOR"]
    )
    # 6. 指标容器
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "current_epoch": 0}
    # 7. 启动训练（传入model_cfg）
    trainer = Trainer(args=args, model_cfg=model_cfg, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, metrics=metrics)
    trainer.train()
