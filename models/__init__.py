import inspect
from models.ViT import vit_cifar10
from models.ResNet import resnet18

# 模型注册表
MODEL_REGISTRY = {
    "vit": vit_cifar10,
    "resnet": resnet18,
}

def build_model(model_name, **kwargs):
    """统一模型构建接口，自动过滤模型不需要的参数"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}")
    
    model_cls = MODEL_REGISTRY[model_name]
    # 只保留模型构造函数支持的参数
    sig = inspect.signature(model_cls)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return model_cls(**valid_kwargs)
