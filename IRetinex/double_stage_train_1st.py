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
    save_dir = "checkpoints/phase1"
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
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
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

            # 3. 计算两个R_init的L1损失（作为第一阶段唯一训练损失）
            loss = nn.functional.l1_loss(R_init_low, R_init_gt)
            # -----------------------------------------------------------------------------

            # 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 损失统计 + 进度条更新
            epoch_loss += loss.item()
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

    logging.info("第一阶段训练完成！")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()