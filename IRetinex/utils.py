import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_model(model, path, epoch=None):
    """保存模型权重"""
    state = {
        'model_state': model.state_dict(),
        'epoch': epoch
    }
    torch.save(state, path)
    print(f"Model saved to {path}")

def load_model(model, path, device='cuda'):
    """加载模型权重"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model loaded from {path}")
    return model, checkpoint.get('epoch', 0)

def visualize_results(original, enhanced, target, save_path=None):
    """可视化结果"""
    # 转换为numpy (HWC)
    def tensor_to_np(tensor):
        return tensor.cpu().numpy().transpose(1, 2, 0)
    
    # 限制在0-1范围内
    original = np.clip(tensor_to_np(original), 0, 1)
    enhanced = np.clip(tensor_to_np(enhanced), 0, 1)
    target = np.clip(tensor_to_np(target), 0, 1)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title('Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(target)
    axes[2].set_title('Target')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()