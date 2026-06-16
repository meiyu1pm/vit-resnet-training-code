import importlib

_CONFIG_REGISTRY = {
    "vit": "configs.vit_config",
    "resnet": "configs.resnet_config",
}

def get_model_config(model_name: str) -> dict:
    """根据模型名动态加载对应配置，返回配置字典"""
    if model_name not in _CONFIG_REGISTRY:
        raise ValueError(f"不支持的模型配置: {model_name}，可选：{list(_CONFIG_REGISTRY.keys())}")
    
    cfg_module = importlib.import_module(_CONFIG_REGISTRY[model_name])
    return {
        k: getattr(cfg_module, k)
        for k in dir(cfg_module)
        if not k.startswith("_")
    }
