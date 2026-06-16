import torch
import os
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None, device=torch.device('cpu')):
    model.train()
    total_loss = 0.0
    correct, total_samples = 0, 0
    progress_bar = tqdm(dataloader, desc="训练中")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # 累计指标
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        progress_bar.set_postfix({
            "loss": total_loss / total_samples,
            "acc": (correct / total_samples) * 100
        })

    avg_loss = total_loss / total_samples
    avg_acc = (correct / total_samples) * 100.0
    progress_bar.close()
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device=torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    correct, total_samples = 0, 0
    progress_bar = tqdm(dataloader, desc="评估中")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            "loss": total_loss / total_samples,
            "acc": (correct / total_samples) * 100
        })

    avg_loss = total_loss / total_samples
    avg_acc = (correct / total_samples) * 100.0
    progress_bar.close()
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, epoch, best_val_acc, save_path: str):
    """原子保存（防止断点损坏）"""
    temp_path = save_path + ".tmp"
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "torch_version": torch.__version__
        }
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, save_path)  # 原子替换
        print(f"✅ 断点保存至 {save_path}")
    except Exception as e:
        print(f"❌ 保存断点失败: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_checkpoint(model, optimizer, load_path: str, device=torch.device('cpu')):
    try:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        return start_epoch, best_val_acc
    except Exception as e:
        raise RuntimeError(f"加载断点失败: {e}") from e
