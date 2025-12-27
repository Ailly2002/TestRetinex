import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import IRetinex
from utils import load_model, visualize_results

def test(model_path, input_path, output_path):
    """测试模型并保存结果"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = IRetinex().to(device)
    model, _ = load_model(model, model_path, device)
    model.eval()
    
    # 读取图像
    image = Image.open(input_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        enhanced, _, _ = model(image_tensor)
    
    # 保存结果
    enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced_np = np.clip(enhanced_np, 0, 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_np)
    plt.title('Enhanced')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Enhanced image saved to {output_path}")

if __name__ == '__main__':
    # 测试示例
    test(
        model_path='checkpoints/iretinex_epoch_50.pth',
        input_path='data/test/test_image.jpg',
        output_path='results/enhanced_test.jpg'
    )