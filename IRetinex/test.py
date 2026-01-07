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
import warnings
from skimage.metrics import structural_similarity as ssim
import math
import traceback


# -------------------------- 工具函数（新增） --------------------------
def resize_tensor_to_match(src_tensor, target_tensor):
    """
    将源张量resize到目标张量的尺寸（保持C维度，调整H/W）
    :param src_tensor: 待调整张量 [C, H, W]
    :param target_tensor: 目标尺寸张量 [C, H, W]
    :return: 尺寸对齐后的源张量 [C, H_target, W_target]
    """
    # 获取目标尺寸
    target_h, target_w = target_tensor.shape[1], target_tensor.shape[2]
    # 构建resize变换（只调整H/W，保持C）
    resize_transform = transforms.Resize((target_h, target_w), antialias=True)
    # 增加batch维度 -> resize -> 去除batch维度
    resized_tensor = resize_transform(src_tensor.unsqueeze(0)).squeeze(0)
    return resized_tensor


# -------------------------- 指标计算函数 --------------------------
def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    :param img1: 增强图像 [C, H, W]，值域0-1
    :param img2: GT图像 [C, H, W]，值域0-1
    :return: PSNR值
    """
    # 前置检查：确保尺寸一致
    if img1.shape != img2.shape:
        print(f"警告：图像尺寸不匹配 {img1.shape} vs {img2.shape}，自动对齐尺寸")
        img1 = resize_tensor_to_match(img1, img2)

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
    # 前置检查：确保尺寸一致
    if img1.shape != img2.shape:
        print(f"警告：图像尺寸不匹配 {img1.shape} vs {img2.shape}，自动对齐尺寸")
        img1 = resize_tensor_to_match(img1, img2)

    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    # 修复：multichannel参数已废弃，改用channel_axis=-1（适配新版skimage）
    ssim_val = ssim(img1_np, img2_np, channel_axis=-1, data_range=1.0)
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
    # 前置检查：确保尺寸一致（避免亮度计算偏差）
    if enhanced_img.shape != gt_img.shape:
        print(f"警告：图像尺寸不匹配 {enhanced_img.shape} vs {gt_img.shape}，自动对齐尺寸")
        enhanced_img = resize_tensor_to_match(enhanced_img, gt_img)

    enhanced_alv = calculate_alv(enhanced_img)
    gt_alv = calculate_alv(gt_img)
    mabd = torch.abs(enhanced_alv - gt_alv)
    return mabd


# -------------------------- 核心增强函数 --------------------------
def lowlight(image_path, gt_path, scale_factor, model_path, save_path, device):
    """
    低光照图像增强，返回增强图像、推理时间及中间/最终的 L/R
    :param image_path: 测试图像路径
    :param gt_path: GT图像路径
    :param scale_factor: 缩放因子（用于裁剪）
    :param model_path: 模型权重路径
    :param save_path: 增强图像保存路径
    :param device: 计算设备（cuda/cpu）
    :return: enhanced_img [C, H, W]、gt_img [C, H, W]、推理时间、L_init、R_init、L_final、R_final
    """
    # 1. 读取并预处理测试图像
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight_np = (np.asarray(data_lowlight) / 255.0).astype(np.float32)
    data_lowlight = torch.from_numpy(data_lowlight_np).float()

    # 2. 读取并预处理GT图像（保持尺寸和测试图像一致）
    gt_img_pil = Image.open(gt_path).convert('RGB')
    gt_img_np = (np.asarray(gt_img_pil) / 255.0).astype(np.float32)
    gt_img = torch.from_numpy(gt_img_np).float()

    # 修复：增加尺寸检查，避免裁剪出错（解包错误的核心诱因之一）
    if len(data_lowlight.shape) != 3 or len(gt_img.shape) != 3:
        raise ValueError(
            f"图像维度错误 - 测试图shape: {data_lowlight.shape}, GT图shape: {gt_img.shape} (期望3维[H,W,C])")

    # 3. 裁剪到scale_factor的整数倍（保证模型输入尺寸合法）
    # 优化：同时对齐测试图和GT图的裁剪尺寸，避免初始尺寸不一致
    h = min(data_lowlight.shape[0], gt_img.shape[0]) // scale_factor * scale_factor
    w = min(data_lowlight.shape[1], gt_img.shape[1]) // scale_factor * scale_factor
    # 增加边界检查，避免裁剪尺寸为0
    if h <= 0 or w <= 0:
        raise ValueError(
            f"裁剪后尺寸非法 - h: {h}, w: {w} (scale_factor={scale_factor}, 测试图尺寸={data_lowlight.shape[:2]}, GT图尺寸={gt_img.shape[:2]})")

    data_lowlight = data_lowlight[0:h, 0:w, :]
    gt_img = gt_img[0:h, 0:w, :]  # GT图像同步裁剪

    # 4. 维度转换 [H, W, C] -> [C, H, W]
    data_lowlight = data_lowlight.permute(2, 0, 1).to(device)
    gt_img = gt_img.permute(2, 0, 1).to(device)

    # 5. 加载模型并推理
    IRetinex_net = models.IRetinex().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state' in checkpoint:
        model_weights = checkpoint['model_state']
    else:
        model_weights = checkpoint
    IRetinex_net.load_state_dict(model_weights, strict=False)
    IRetinex_net.eval()

    start = time.time()
    with torch.no_grad():
        model_output = IRetinex_net(data_lowlight.unsqueeze(0))  # 以 batch=1 输入
        # 兼容多种返回形式：优先解析 enhanced, L_init, R_init, L_final, R_final
        if isinstance(model_output, (tuple, list)):
            enhanced_image = model_output[0]
            L_init = model_output[1] if len(model_output) > 1 else None
            R_init = model_output[2] if len(model_output) > 2 else None
            L_final = model_output[3] if len(model_output) > 3 else None
            R_final = model_output[4] if len(model_output) > 4 else None
        else:
            # 兼容单输出模型（旧行为）
            enhanced_image = model_output
            L_init = R_init = L_final = R_final = None
    infer_time = time.time() - start

    # 6. 后处理：去除batch维度 + 强制对齐GT尺寸 + 限制值域
    enhanced_image = enhanced_image.squeeze(0)  # [C, H, W]
    if enhanced_image.shape != gt_img.shape:
        print(f"模型输出尺寸 {enhanced_image.shape} 与GT尺寸 {gt_img.shape} 不一致，自动对齐")
        enhanced_image = resize_tensor_to_match(enhanced_image, gt_img)
    enhanced_image = torch.clamp(enhanced_image, 0, 1)  # 限制值域0-1

    # 7. 保存增强图像（原位置）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image.cpu(), save_path)

    # 8. 保存中间与最终的 L/R 到指定子文件夹（mid / enhanced）
    base_dir = os.path.dirname(save_path)  # 在同一级目录下创建子目录
    mid_dir = os.path.join(base_dir, 'mid')
    enh_dir = os.path.join(base_dir, 'enhanced')
    os.makedirs(mid_dir, exist_ok=True)
    os.makedirs(enh_dir, exist_ok=True)

    img_name = os.path.basename(save_path)
    name_wo_ext = os.path.splitext(img_name)[0]

    # 定义一个保存函数：先对齐到GT尺寸再保存
    def _save_tensor(tensor, target_path):
        if tensor is None:
            return
        t = tensor.squeeze(0) if tensor.dim() == 4 else tensor  # 去除batch
        # 若尺寸与GT不一致则对齐
        if t.shape != gt_img.shape:
            t = resize_tensor_to_match(t, gt_img)
        t = torch.clamp(t, 0, 1)
        torchvision.utils.save_image(t.cpu(), target_path)

    # 保存 L_init / R_init -> mid
    _save_tensor(L_init, os.path.join(mid_dir, f"{name_wo_ext}_L_init.png"))
    _save_tensor(R_init, os.path.join(mid_dir, f"{name_wo_ext}_R_init.png"))

    # 保存 L_final / R_final -> enhanced 子目录（与最终增强图区分）
    _save_tensor(L_final, os.path.join(enh_dir, f"{name_wo_ext}_L_final.png"))
    _save_tensor(R_final, os.path.join(enh_dir, f"{name_wo_ext}_R_final.png"))

    return enhanced_image, gt_img, infer_time, L_init, R_init, L_final, R_final


# -------------------------- 主函数 --------------------------
def main():
    # 1. 解析命令行参数
    args = SimpleNamespace(
        test_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/low',
        gt_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/high',
        save_root=r'E:/Experiences/LOL/IRetinex/RELU',
        # model_path=r'./snapshot/20260105_225353/Epoch_100_20260105_225353.pth',
        model_path = r'checkpoints/phase1/phase1_epoch_30.pth',
        scale_factor=12,
        gpu_id='0'
    )

    # 2. 设备配置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True if torch.cuda.is_available() else False

    # 3. 初始化LPIPS模型（感知相似度计算）
    # 修复：适配旧版LPIPS，使用pretrained参数 + 抑制警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # 抑制pretrained弃用警告
        lpips_model = lpips.LPIPS(net='alex', pretrained=True, spatial=False).to(device)  # 恢复pretrained参数

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
    # 修复：优化glob遍历，避免匹配到非图片文件，增加路径过滤
    test_file_list = glob.glob(os.path.join(args.test_root, '**', '*'), recursive=True)
    test_file_list = [
        f for f in test_file_list
        if os.path.isfile(f) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    print(f"找到 {len(test_file_list)} 张测试图片")

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
            enhanced_img, gt_img, infer_time, L_init, R_init, L_final, R_final = lowlight(
                test_img_path, gt_img_path, args.scale_factor,
                args.model_path, save_img_path, device
            )
        except Exception as e:
            print(f"处理图片 {test_img_path} 时出错：{e}，跳过该图片")
            traceback.print_exc()
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