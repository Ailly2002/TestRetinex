import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import os
from types import SimpleNamespace
import sys
import argparse
import time
import models
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import lpips
from skimage.metrics import structural_similarity as ssim
import math


# -------------------------- 指标计算函数 --------------------------
def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    :param img1: 增强图像 [C, H, W]，值域0-1
    :param img2: GT图像 [C, H, W]，值域0-1
    :return: PSNR值
    """
    img1 = img1.float()
    img2 = img2.float()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse)


def calculate_ssim(img1, img2):
    """
    计算SSIM（结构相似性）
    :param img1: 增强图像 [C, H, W]，值域0-1
    :param img2: GT图像 [C, H, W]，值域0-1
    :return: SSIM值
    """
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    # 多通道SSIM计算，data_range=1.0（值域0-1）
    ssim_val = ssim(img1_np, img2_np, multichannel=True, data_range=1.0)
    return ssim_val


def calculate_alv(img):
    """
    计算ALV（平均亮度值）
    :param img: 图像 [C, H, W]，值域0-1
    :return: 平均亮度值
    """
    img = img.float()
    # RGB转灰度：Y = 0.299R + 0.587G + 0.114B
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    alv = torch.mean(gray)
    return alv


def calculate_mabd(enhanced_img, gt_img):
    """
    计算MABD（平均亮度差异）
    :param enhanced_img: 增强图像 [C, H, W]，值域0-1
    :param gt_img: GT图像 [C, H, W]，值域0-1
    :return: MABD值
    """
    enhanced_alv = calculate_alv(enhanced_img)
    gt_alv = calculate_alv(gt_img)
    mabd = torch.abs(enhanced_alv - gt_alv)
    return mabd


# -------------------------- 核心增强函数 --------------------------
def lowlight(image_path, gt_path, scale_factor, model_path, save_path, device):
    """
    低光照图像增强，返回增强图像、推理时间
    :param image_path: 测试图像路径
    :param gt_path: GT图像路径
    :param scale_factor: 缩放因子（用于裁剪）
    :param model_path: 模型权重路径
    :param save_path: 增强图像保存路径
    :param device: 计算设备（cuda/cpu）
    :return: enhanced_img [C, H, W]、gt_img [C, H, W]、推理时间
    """
    # 1. 读取并预处理测试图像
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight_np = (np.asarray(data_lowlight) / 255.0).astype(np.float32)
    data_lowlight = torch.from_numpy(data_lowlight_np).float()

    # 2. 读取并预处理GT图像（保持尺寸和测试图像一致）
    gt_img_pil = Image.open(gt_path).convert('RGB')
    gt_img_np = (np.asarray(gt_img_pil) / 255.0).astype(np.float32)
    gt_img = torch.from_numpy(gt_img_np).float()

    # 3. 裁剪到scale_factor的整数倍（保证模型输入尺寸合法）
    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    gt_img = gt_img[0:h, 0:w, :]  # GT图像同步裁剪

    # 4. 维度转换 [H, W, C] -> [C, H, W]
    data_lowlight = data_lowlight.permute(2, 0, 1).to(device)
    gt_img = gt_img.permute(2, 0, 1).to(device)

    # 5. 加载模型并推理
    IRetinex_net = models.IRetinex().to(device)
    # 加载权重文件（包含model_state、epoch等）
    checkpoint = torch.load(model_path, map_location=device)
    # 提取真正的模型state_dict（从model_state键中）
    if 'model_state' in checkpoint:
        model_weights = checkpoint['model_state']
    else:
        model_weights = checkpoint  # 兼容直接保存state_dict的情况
    # 加载权重，设置strict=False忽略少量不匹配（如果模型结构略有差异）
    IRetinex_net.load_state_dict(model_weights, strict=False)
    # 设置模型为评估模式
    IRetinex_net.eval()

    start = time.time()
    with torch.no_grad():
        enhanced_image, _ = IRetinex_net(data_lowlight.unsqueeze(0))  # [1, C, H, W]
    infer_time = time.time() - start

    # 6. 后处理：去除batch维度，裁剪到GT尺寸（防止模型输出尺寸偏移）
    enhanced_image = enhanced_image.squeeze(0)  # [C, H, W]
    enhanced_image = torch.clamp(enhanced_image, 0, 1)  # 限制值域0-1

    # 7. 保存增强图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, save_path)

    return enhanced_image, gt_img, infer_time


# -------------------------- 主函数 --------------------------
def main():
    # 1. 解析命令行参数
    # parser = argparse.ArgumentParser(description='Zero-DCE++ Test with Metrics')
    # parser.add_argument('--test_root', type=str, required=True, help='测试图片根路径')
    # parser.add_argument('--gt_root', type=str, required=True, help='GT图片根路径')
    # parser.add_argument('--save_root', type=str, required=True, help='增强结果保存根路径')
    # parser.add_argument('--model_path', type=str, default='snapshots_Zero_DCE++/Epoch99.pth', help='模型权重路径')
    # parser.add_argument('--scale_factor', type=int, default=12, help='图像裁剪缩放因子')
    # parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID（如0或0,1）')
    # args = parser.parse_args()

    # 配置：直接在此修改变量，替代命令行参数
    # 修改路径请保持使用反斜杠或原始字符串（Windows）如 `r"C:\data\test"`
    args = SimpleNamespace(
        test_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/low',
        gt_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/high',
        save_root=r'E:/Experiences/LOL/IRetinex/20260105_225353',
        model_path=r'./snapshot/20260105_225353/Epoch_100_20260105_225353.pth',
        scale_factor=12,
        gpu_id='0'
    )

    # 2. 设备配置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True if torch.cuda.is_available() else False

    # 3. 初始化LPIPS模型（感知相似度计算）
    lpips_model = lpips.LPIPS(net='alex').to(device)  # 使用AlexNet作为backbone

    # 4. 初始化指标累加器
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0
    sum_enhanced_alv = 0.0
    sum_gt_alv = 0.0
    sum_mabd = 0.0
    sum_infer_time = 0.0
    img_count = 0

    # 5. 遍历所有测试图片
    test_file_list = glob.glob(os.path.join(args.test_root, '**/*.*'), recursive=True)
    test_file_list = [f for f in test_file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for test_img_path in test_file_list:
        # 匹配GT图片路径（假设文件名相同，路径不同）
        img_name = os.path.basename(test_img_path)
        gt_img_path = os.path.join(args.gt_root, img_name)
        if not os.path.exists(gt_img_path):
            print(f"警告：GT图片 {gt_img_path} 不存在，跳过该图片")
            continue

        # 构建增强结果保存路径
        rel_path = os.path.relpath(test_img_path, args.test_root)
        save_img_path = os.path.join(args.save_root, rel_path)

        # 低光照增强
        try:
            enhanced_img, gt_img, infer_time = lowlight(
                test_img_path, gt_img_path, args.scale_factor,
                args.model_path, save_img_path, device
            )
        except Exception as e:
            print(f"处理图片 {test_img_path} 时出错：{e}，跳过该图片")
            continue

        # 计算指标
        psnr = calculate_psnr(enhanced_img, gt_img)
        ssim_val = calculate_ssim(enhanced_img, gt_img)
        lpips_val = lpips_model(enhanced_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
        enhanced_alv = calculate_alv(enhanced_img).item()
        gt_alv = calculate_alv(gt_img).item()
        mabd = calculate_mabd(enhanced_img, gt_img).item()

        # 累加指标
        sum_psnr += psnr
        sum_ssim += ssim_val
        sum_lpips += lpips_val
        sum_enhanced_alv += enhanced_alv
        sum_gt_alv += gt_alv
        sum_mabd += mabd
        sum_infer_time += infer_time
        img_count += 1

        # 打印单张图片指标
        print(f"图片 {img_name}:")
        print(f"  PSNR: {psnr:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
        print(f"  增强图ALV: {enhanced_alv:.4f}, GT图ALV: {gt_alv:.4f}, MABD: {mabd:.4f}")
        print(f"  推理时间: {infer_time:.4f}s\n")

    # 6. 输出平均指标
    if img_count == 0:
        print("未处理任何有效图片！")
        return

    avg_psnr = sum_psnr / img_count
    avg_ssim = sum_ssim / img_count
    avg_lpips = sum_lpips / img_count
    avg_enhanced_alv = sum_enhanced_alv / img_count
    avg_gt_alv = sum_gt_alv / img_count
    avg_mabd = sum_mabd / img_count
    avg_infer_time = sum_infer_time / img_count
    total_infer_time = sum_infer_time

    print("=" * 50)
    print("测试汇总（平均指标）：")
    print(f"处理图片总数: {img_count}")
    print(f"平均PSNR: {avg_psnr:.4f}")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"平均LPIPS: {avg_lpips:.4f}")
    print(f"增强图平均ALV: {avg_enhanced_alv:.4f}")
    print(f"GT图平均ALV: {avg_gt_alv:.4f}")
    print(f"平均MABD: {avg_mabd:.4f}")
    print(f"平均推理时间: {avg_infer_time:.4f}s/张")
    print(f"总推理时间: {total_infer_time:.4f}s")
    print("=" * 50)


if __name__ == '__main__':
    with torch.no_grad():
        main()