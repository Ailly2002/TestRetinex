import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np

# 导入核心模块（与你的train.py保持一致）
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
    seed = 42
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

    # 加载数据集（现在在 main 内，避免 spawn 时重复导入执行）
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

    # 模型初始化 + 参数冻结
    model = IRetinex().to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.dual_color.parameters():
        param.requires_grad = True
    for param in model.reflectance_decomp.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss().to(device)

    # 训练循环
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for batch_idx, (low_light_imgs, gt_imgs) in enumerate(pbar):
            low_light_imgs = low_light_imgs.to(device, dtype=torch.float32)
            gt_imgs = gt_imgs.to(device, dtype=torch.float32)

            L_init = model.dual_color(low_light_imgs)
            R_init = model.reflectance_decomp(low_light_imgs, L_init)

            loss = criterion(R_init, gt_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})

        save_path = os.path.join(save_dir, f"phase1_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, save_path)
        logging.info(f"Epoch {epoch} 保存至 {save_path} | 平均损失：{avg_loss:.6f}")

    logging.info("第一阶段训练完成！")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
