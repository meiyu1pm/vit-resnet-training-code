import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from configs.basic_config import NUM_WORKERS

def _load_single_batch(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f, encoding='latin1')
        return np.array(batch_data["data"]), np.array(batch_data["labels"])
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件{file_path}不存在, 请检查DATA_ROOT")
    except Exception as e:
        raise RuntimeError(f"加载批次{file_path}失败: {e}") from e

class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = img.reshape((3, 32, 32)).transpose((1, 2, 0)).astype(np.uint8) # (CH, H, W) -> (H, W, CH)
        if self.transform:
            img = self.transform(img)
        return img, label

def build_train_val_loader(
        train_batch_list: list,
        val_size: int,
        train_size: int,
        batch_size: int = 64,
        num_workers: int = NUM_WORKERS,
        img_size: int = 32):
    image_list, label_list = [], []
    for file_path in train_batch_list:
        imgs, labs = _load_single_batch(file_path)
        image_list.append(imgs)
        label_list.append(labs)
    
    all_images = np.concatenate(image_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    
    # 打乱（固定种子）
    idx = np.arange(len(all_images))
    random.shuffle(idx)
    all_images = all_images[idx]
    all_labels = all_labels[idx]
    
    # 划分
    train_images = all_images[:train_size]
    train_labels = all_labels[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    val_labels = all_labels[train_size:train_size + val_size]

    # 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 新增随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 新增颜色抖动
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_images, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 避免最后一个batch尺寸不一致
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    return train_loader, val_loader

def build_test_loader(
        test_batch_path: str,
        batch_size: int = 64,
        num_workers: int = NUM_WORKERS,
        img_size: int = 32,
        pin_memory=True):
    test_images, test_labels = _load_single_batch(test_batch_path)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    print(f"测试集大小: {len(test_dataset)}")
    return test_loader
