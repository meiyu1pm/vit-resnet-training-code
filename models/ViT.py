import torch
import torch.nn as nn
from configs import vit_config
from configs.vit_config import *

"""
Vision Transformer Module
copyed from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

# DropPath: 
def drop_path(x, drop_prob: float=0., training: bool=False):
    # x: Tensor [N, C, H, W]
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape: (N, 1, 1, 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# PatchEmbed:
class PatchEmbed(nn.Module):
    def __init__(self, 
                 img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_c=IN_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [N, C, H, W]
        x = self.proj(x)  # 提取patch [N, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [N, H*W, embed_dim]
        return x


# Multi-Head Attention:
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, # EMBED_DIM,是patch embedding的输出维度
                 num_heads:int=NUM_HEADS, 
                 qkv_bias=True, qkv_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qkv_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim) # Linear Projection of Flattened Patches
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape # Batch Size, Number of Patches, Embedding Dimension
        qkv = self.qkv(x).reshape( # [B, N, 3, self.num_heads, C // self.num_heads])
            B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, self.num_heads, N, C // self.num_heads]
        q, k, v = qkv.unbind(0) # [B, self.num_heads, N, C // self.num_heads]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, self.num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, N, C]
        x = self.proj_drop(self.proj(x)) 
        return x


# MLP 前馈网络
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

# TransformerEncoder 一个TransformerEncoderBlock
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qkv_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qkv_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # 两个残差连接：注意力 + MLP
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

# VistionTransformer Block
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_c=IN_CHANNELS, num_classes=NUM_CLASSES,
                 embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS, mlp_ratio=4,
                 qkv_bias=False, qkv_scale=None,
                 drop_ratio=DROP_RATIO, attn_drop_ratio=ATTN_DROP_RATIO, drop_path_ratio=DROP_PATH_RATIO,
                 embed_layer=PatchEmbed, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim

        # 内置归一化无梯度buffer
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        )


        # 1. Patch嵌入
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. 可学习的分类token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 3. 堆叠Transformer编码器
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            TransformerEncoder(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qkv_scale=qkv_scale,
                               drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                               drop_path_ratio=dpr[i],
                               norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 4. 唯一的分类头：一个线性层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # 归一化操作, x: [B, C, H, W]
        x = (x - self.mean) / self.std
        # 图片切块patch
        x = self.patch_embed(x)
        # 拼接分类token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 加位置编码
        x = self.pos_drop(x + self.pos_embed)
        # 过所有Transformer层
        x = self.blocks(x)
        x = self.norm(x)
        # 只取第一个位置的分类token输出
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# 对外唯一接口：构造适配CIFAR-10的VisionTransformer模型
def vit_cifar10(**kwargs):
    """
    构造适配CIFAR-10的VisionTransformer模型
    默认参数基于configs/vit_config.py 支持通过kwargs覆盖任意参数
    
    Args:
        **kwargs: 可覆盖的参数如img_size、patch_size、embed_dim、depth、num_heads等
    Returns:
        VisionTransformer: 初始化完成的ViT模型
    """
    # 基础默认参数（和vit_config.py对齐）
    default_kwargs = {
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "in_c": IN_CHANNELS,
        "num_classes": NUM_CLASSES,
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_ratio": 4,
        "qkv_bias": QKV_BIAS if hasattr(vit_config, "QKV_BIAS") else False,
        "qkv_scale": None,
        "drop_ratio": DROP_RATIO,
        "attn_drop_ratio": ATTN_DROP_RATIO,
        "drop_path_ratio": DROP_PATH_RATIO,
        "act_layer": nn.GELU,
        "norm_layer": nn.LayerNorm
    }
    # 用kwargs覆盖默认参数（优先级更高）
    default_kwargs.update(kwargs)
    
    # 构造并返回模型
    return VisionTransformer(**default_kwargs)
