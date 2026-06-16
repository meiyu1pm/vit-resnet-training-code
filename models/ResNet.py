import torch
import torch.nn as nn
from configs.resnet_config import *

"""
ResNet-18 适配 CIFAR-10 (32×32 输入)
仅包含模型结构定义，训练/数据/断点全部复用项目公共工具
"""

# 残差基础块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 两个 3×3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 短路连接
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ResNet 主网络
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 img_size=IMG_SIZE,
                 in_c=IN_CHANNELS,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = 64

        # 内置归一化（和 ViT 完全一致，保证输入数据 100% 共用、对比公平）
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        )

        # CIFAR-10 适配：替换 7×7 大卷积为 3×3，取消最大池化，避免特征图过早缩小
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 组残差块
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self.apply(_init_resnet_weights)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 维度不匹配时，用 1×1 卷积做短路连接
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        # 前置归一化（和 ViT 逻辑完全对齐）
        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


# 权重初始化（ResNet 标准实现）
def _init_resnet_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)


# 对外唯一接口：构造 ResNet-18
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
