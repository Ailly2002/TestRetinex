import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np

from models.main_model import IRetinex
from data_loader import RetinexDataset  # 你的数据加载类

# 新增：用于绘制训练曲线的简单函数（Loss, PSNR, SSIM）
import matplotlib.pyplot as plt
import datetime
from utils import calculate_psnr, calculate_ssim  # 假定项目已有 utils 中实现

def plot_training_metrics(epoch_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = [m['epoch'] for m in epoch_metrics]
    losses = [m['loss'] for m in epoch_metrics]
    psnrs = [m['psnr'] for m in epoch_metrics]
    ssims = [m['ssim'] for m in epoch_metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, '-o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnrs, '-o', label='PSNR (dB)')
    plt.plot(epochs, ssims, '-o', label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'psnr_ssim_curve.png'))
    plt.close()

def main():
    # ====================== 基础参数设置 ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_root = "E:/Low-LightDatasets/Images/LOLdataset/our485/low"
    gt_root = "E:/Low-LightDatasets/Images/LOLdataset/our485/high"
    batch_size = 8
    img_size = 512
    lr = 1e-4
    epochs = 30
    weight_decay = 1e-5
    num_workers = 4 if torch.cuda.is_available() else 0
    # save_dir = "checkpoints/phase1"
    save_dir = "checkpoints/phase1_351"
    os.makedirs(save_dir, exist_ok=True)
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # 加载数据集（低光Input + 高光GT成对样本）
    train_dataset = RetinexDataset(
        input_root=input_root,
        gt_root=gt_root,
        size=img_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"训练集加载完成：{len(train_dataset)} 对样本 | {len(train_loader)} 个批次")

    # 模型初始化 + 参数冻结（仅解冻dual_color和reflectance_decomp）
    model = IRetinex().to(device)
    for param in model.parameters():
        param.requires_grad = False  # 冻结所有参数
    # 仅解冻光照分支和反射率分解分支（核心训练模块）
    for param in model.dual_color.parameters():
        param.requires_grad = True
    for param in model.reflectance_decomp.parameters():
        param.requires_grad = True

    # 优化器：仅更新解冻的参数
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    # 训练循环
    model.train()

    # 新增：用于记录每轮指标
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        sample_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for batch_idx, (low_light_imgs, gt_imgs) in enumerate(pbar):
            # 数据预处理：移至指定设备 + 转换精度
            low_light_imgs = low_light_imgs.to(device, dtype=torch.float32)
            gt_imgs = gt_imgs.to(device, dtype=torch.float32)

            # ---------------------- 核心修改：分别计算Input和GT的R_init ----------------------
            # 1. 低光Input经模型分解得到 R_init_low
            L_init_low = model.dual_color(low_light_imgs)
            R_init_low = model.reflectance_decomp(low_light_imgs, L_init_low)

            # 2. GT图经模型分解得到 R_init_gt
            L_init_gt = model.dual_color(gt_imgs)
            R_init_gt = model.reflectance_decomp(gt_imgs, L_init_gt)

            # 3. 计算损失：保留原有 R_init 之间的 L1，同时新增 I_init 的重建损失
            #    I_init = R_init * L_init，与对应输入图像比较
            # 兼容通道不匹配：若 L 为单通道则广播，否则尝试裁剪/重复以匹配 R 的通道数
            def _align_and_mul(R, L):
                if L.shape[1] != R.shape[1]:
                    if L.shape[1] == 1:
                        L = L.repeat(1, R.shape[1], 1, 1)
                    elif R.shape[1] == 1:
                        R = R.repeat(1, L.shape[1], 1, 1)
                    else:
                        # 最简单的处理：如果通道不匹配，按最小通道数裁剪多余通道
                        min_c = min(R.shape[1], L.shape[1])
                        R = R[:, :min_c, ...]
                        L = L[:, :min_c, ...]
                return R * L

            I_init_low = _align_and_mul(R_init_low, L_init_low)
            I_init_gt = _align_and_mul(R_init_gt, L_init_gt)

            # 原有的 R_init 之间的 L1 损失
            loss_R = nn.functional.l1_loss(R_init_low, R_init_gt)

            # 新增的 I_init 与对应输入图像的 L1 损失
            loss_I_low = nn.functional.l1_loss(I_init_low, low_light_imgs)
            loss_I_gt = nn.functional.l1_loss(I_init_gt, gt_imgs)

            # 最终第一阶段损失：三项相加
            loss = loss_R + loss_I_low + loss_I_gt
            # -----------------------------------------------------------------------------

            # 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 损失统计 + 进度条更新
            epoch_loss += loss.item()

            # 评估指标统计：使用 I_init_gt 与 GT 计算 PSNR/SSIM（也可使用 I_init_low 与 low_light）
            # 将单张图转换为 numpy [0,255,H,W->H,W,C] 供 utils 函数使用
            with torch.no_grad():
                I_gt_np = (I_init_gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.float32)
                gt_np = (gt_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.float32)
                batch_psnr = 0.0
                batch_ssim = 0.0
                bs = I_gt_np.shape[0]
                for b in range(bs):
                    batch_psnr += calculate_psnr(I_gt_np[b], gt_np[b])
                    batch_ssim += calculate_ssim(I_gt_np[b], gt_np[b])
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
                sample_count += bs

            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.6f}",
                "avg_loss": f"{avg_loss:.6f}"
            })

        # 每轮训练结束后保存模型权重
        save_path = os.path.join(save_dir, f"phase1_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss
        }, save_path)
        logging.info(f"Epoch {epoch} 保存至 {save_path} | 本轮平均L1损失：{avg_loss:.6f}")

        # 计算并记录本轮平均指标（PSNR/SSIM 按像素样本数归一）
        if sample_count > 0:
            avg_epoch_psnr = epoch_psnr / sample_count
            avg_epoch_ssim = epoch_ssim / sample_count
        else:
            avg_epoch_psnr = 0.0
            avg_epoch_ssim = 0.0

        epoch_metrics.append({
            'epoch': epoch,
            'loss': avg_loss,
            'psnr': avg_epoch_psnr,
            'ssim': avg_epoch_ssim
        })

        # 每轮结束时绘图并保存（逐轮更新图像）
        try:
            plot_training_metrics(epoch_metrics, save_dir)
        except Exception as e:
            logging.warning(f"Plotting failed at epoch {epoch}: {e}")

    logging.info("第一阶段训练完成！")
    # 最后再保存一次完整的曲线图
    try:
        plot_training_metrics(epoch_metrics, save_dir)
    except Exception:
        pass


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()