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
import torch.nn.functional as F
import math
import traceback


# -------------------------- 工具函数（新增） --------------------------
def resize_tensor_to_match(src_tensor, target_tensor):
    """
    将 src_tensor 调整到 target_tensor 的 H/W（保持 C）。
    支持输入为 [C,H,W], [H,W,C], [1,C,H,W] 等常见格式。
    返回 [C, H_target, W_target] 的 tensor（与 target_tensor 在同一设备）。
    """
    t = src_tensor
    # 去除 batch 维
    if t.dim() == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    # 如果是 HWC -> CHW
    if t.dim() == 3 and t.shape[2] in (1, 3) and t.shape[0] not in (1, 3):
        t = t.permute(2, 0, 1)
    if t.dim() != 3:
        raise ValueError(f"无法识别的张量维度用于 resize: {src_tensor.shape}")
    target_h, target_w = target_tensor.shape[1], target_tensor.shape[2]
    t = t.unsqueeze(0)  # [1,C,H,W]
    # 使用双线性插值（保持 float）
    t_resized = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return t_resized.squeeze(0)


# -------------------------- 指标计算函数 --------------------------
def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    :param img1: 增强图像 [C, H, W]，值域0-1
    :param img2: GT图像 [C, H, W]，值域0-1
    :return: PSNR值
    """
    if img1.shape != img2.shape:
        print(f"警告：图像尺寸不匹配 {img1.shape} vs {img2.shape}，自动对齐尺寸")
        img1 = resize_tensor_to_match(img1, img2)
    img1 = img1.float()
    img2 = img2.float()
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)


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
    低光照图像增强（修正版）
    - 对输入先裁到两图共同最小尺寸，再 pad 到 align 的倍数（reflect），
      避免多尺度特征图尺寸不匹配导致的 runtime error。
    """
    # 1. 读取并预处理测试图像
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight_np = (np.asarray(data_lowlight) / 255.0).astype(np.float32)
    data_lowlight = torch.from_numpy(data_lowlight_np).float()

    # 2. 读取并预处理GT图像（保持尺寸和测试图像一致）
    gt_img_pil = Image.open(gt_path).convert('RGB')
    gt_img_np = (np.asarray(gt_img_pil) / 255.0).astype(np.float32)
    gt_img = torch.from_numpy(gt_img_np).float()

    # 前置检查
    if len(data_lowlight.shape) != 3 or len(gt_img.shape) != 3:
        raise ValueError(
            f"图像维度错误 - 测试图shape: {data_lowlight.shape}, GT图shape: {gt_img.shape} (期望3维[H,W,C])")

    # --- 改动开始：先裁到两张图的共同最小尺寸，再 pad 到 align 的倍数 ---
    # 取共同最小尺寸，避免直接用不同尺寸造成对齐问题
    base_h = min(data_lowlight.shape[0], gt_img.shape[0])
    base_w = min(data_lowlight.shape[1], gt_img.shape[1])

    # 裁切到共同最小尺寸（避免后续 pad 导致 GT/测试图尺寸不一致）
    data_lowlight = data_lowlight[0:base_h, 0:base_w, :]
    gt_img = gt_img[0:base_h, 0:base_w, :]

    # 对齐到某个 \`align\` 倍数（选择 16 可避免多数下采样/上采样带来的奇偶差）
    align = 16
    target_h = int(math.ceil(base_h / align) * align)
    target_w = int(math.ceil(base_w / align) * align)

    # 如果需要 pad，则在后边和右边进行反射填充（保持内容不突兀）
    pad_h = target_h - base_h
    pad_w = target_w - base_w

    # 转为 CHW，移动到 device 之前做 pad（F.pad 支持 CHW）
    data_lowlight = data_lowlight.permute(2, 0, 1)  # [C, H, W]
    gt_img = gt_img.permute(2, 0, 1)  # [C, H, W]

    if pad_h > 0 or pad_w > 0:
        # F.pad pad 格式： (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
        # 这里只在右和下方填充
        padding = (0, pad_w, 0, pad_h)
        data_lowlight = F.pad(data_lowlight.unsqueeze(0), padding, mode='reflect').squeeze(0)
        gt_img = F.pad(gt_img.unsqueeze(0), padding, mode='reflect').squeeze(0)

    # 移动到设备
    data_lowlight = data_lowlight.to(device)
    gt_img = gt_img.to(device)
    # --- 改动结束 ---

    # 5. 载入模型并推理（其余逻辑与原实现一致）
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
        if isinstance(model_output, (tuple, list)):
            enhanced_image = model_output[0]
            L_init = model_output[1] if len(model_output) > 1 else None
            R_init = model_output[2] if len(model_output) > 2 else None
            L_final = model_output[3] if len(model_output) > 3 else None
            R_final = model_output[4] if len(model_output) > 4 else None
        else:
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
    base_dir = os.path.dirname(save_path)
    mid_dir = os.path.join(base_dir, 'mid')
    enh_dir = os.path.join(base_dir, 'enhanced')
    os.makedirs(mid_dir, exist_ok=True)
    os.makedirs(enh_dir, exist_ok=True)

    img_name = os.path.basename(save_path)
    name_wo_ext = os.path.splitext(img_name)[0]

    def _save_tensor(tensor, target_path, gt_img=None):
        if tensor is None:
            return
        t = tensor
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        if gt_img is not None and t.shape != gt_img.shape:
            t = resize_tensor_to_match(t, gt_img)
        t = torch.clamp(t, 0.0, 1.0).cpu()
        torchvision.utils.save_image(t, target_path)

    _save_tensor(L_init, os.path.join(mid_dir, f"{name_wo_ext}_L_init.png"), gt_img=gt_img)
    _save_tensor(R_init, os.path.join(mid_dir, f"{name_wo_ext}_R_init.png"), gt_img=gt_img)
    if L_init is not None and R_init is not None:
        try:
            I_mid = torch.clamp(R_init * L_init, 0.0, 1.0)
            _save_tensor(I_mid, os.path.join(mid_dir, f"{name_wo_ext}_I_mid.png"), gt_img=gt_img)
        except Exception:
            L_aligned = resize_tensor_to_match(L_init, gt_img) if L_init is not None and L_init.shape != gt_img.shape else L_init
            R_aligned = resize_tensor_to_match(R_init, gt_img) if R_init is not None and R_init.shape != gt_img.shape else R_init
            if L_aligned is not None and R_aligned is not None:
                I_mid = torch.clamp(R_aligned * L_aligned, 0.0, 1.0)
                _save_tensor(I_mid, os.path.join(mid_dir, f"{name_wo_ext}_I_mid.png"), gt_img=gt_img)

    _save_tensor(L_final, os.path.join(enh_dir, f"{name_wo_ext}_L_final.png"))
    _save_tensor(R_final, os.path.join(enh_dir, f"{name_wo_ext}_R_final.png"))

    return enhanced_image, gt_img, infer_time, L_init, R_init, L_final, R_final


# -------------------------- 主函数 --------------------------
def main():
    # 1. 解析命令行参数
    args = SimpleNamespace(
        test_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/low',
        gt_root=r'E:/Low-LightDatasets/Images/LOLdataset/eval15/high',
        save_root=r'E:/Experiences/LOL/IRetinex/20260108_225550-Epoch100',
        # model_path=r'./snapshot/20260105_225353/Epoch_100_20260105_225353.pth',
        model_path = r'./snapshot/20260108_225550/Epoch_100_20260108_225550.pth',
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
        inp_enh = (enhanced_img.unsqueeze(0).to(device) * 2.0 - 1.0)
        inp_gt = (gt_img.unsqueeze(0).to(device) * 2.0 - 1.0)
        lpips_val = lpips_model(inp_enh, inp_gt).item()
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