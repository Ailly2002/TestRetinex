import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

def calculate_psnr(pred, gt):
    """计算单张图像的PSNR（输入为0-255的numpy数组，RGB格式）"""
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    return psnr(gt, pred, data_range=255)

def calculate_ssim(pred, gt):
    """计算单张图像的SSIM（输入为0-255的numpy数组，RGB格式）"""
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    return ssim(gt, pred, data_range=255, channel_axis=-1)