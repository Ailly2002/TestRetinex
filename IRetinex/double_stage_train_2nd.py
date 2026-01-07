import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from models import IRetinex, MultiScaleConsistencyLoss
import data_loader
from utils import save_model, visualize_results, calculate_psnr, calculate_ssim, load_model
import datetime


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_phase1_weights(model, phase1_ckpt_path, device):
    """加载第一阶段ICRR模块权重"""
    checkpoint = torch.load(phase1_ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        phase1_state_dict = checkpoint["model_state_dict"]
    else:
        phase1_state_dict = checkpoint

    icrr_state_dict = {}
    for k, v in phase1_state_dict.items():
        if k.startswith("dual_color.") or k.startswith("reflectance_decomp."):
            icrr_state_dict[k] = v

    model.load_state_dict(icrr_state_dict, strict=False)
    print(f"✅ 成功加载第一阶段ICRR模块权重：{phase1_ckpt_path}")
    print(f"🔍 加载的参数数量：{len(icrr_state_dict)}")
    return model


def train(config):
    # 时间戳与快照目录
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join("snapshot_phase2", timestamp)
    os.makedirs(snapshot_dir, exist_ok=True)

    # 日志文件
    log_file = os.path.join(snapshot_dir, f"training_log_phase2_{timestamp}.txt")
    with open(log_file, 'w') as f:
        f.write(f"===== 第二阶段训练配置 =====\n")
        f.write(f"Training Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Phase1 Checkpoint: {config.phase1_ckpt}\n")
        f.write(f"Data Root: Input={config.input_root}, GT={config.gt_root}\n")
        f.write(f"Image Size: {config.image_size}\n")
        f.write(f"Learning Rate: {config.lr}\n")
        f.write(f"Weight Decay: {config.weight_decay}\n")
        f.write(f"Gradient Clip Norm: {config.grad_clip_norm}\n")
        f.write(f"Number of Epochs: {config.num_epochs}\n")
        f.write(f"Batch Size: {config.train_batch_size}\n")
        f.write(f"Number of Workers: {config.num_workers}\n")
        f.write(f"Display Iter: {config.display_iter}\n")
        f.write(f"Snapshot Iter: {config.snapshot_iter}\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = True

    # 初始化模型 + 加载第一阶段权重
    retinex_net = IRetinex().cuda()
    retinex_net.apply(weights_init)

    if os.path.exists(config.phase1_ckpt):
        retinex_net = load_phase1_weights(retinex_net, config.phase1_ckpt, "cuda")
    else:
        raise ValueError(f"❌ 第一阶段权重文件不存在：{config.phase1_ckpt}")

    # 确认所有参数可训练
    trainable_params = sum(p.numel() for p in retinex_net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in retinex_net.parameters())
    print(f"📊 模型参数统计：总参数={total_params:,} | 可训练参数={trainable_params:,}（全部可训练）")

    # 加载数据集
    train_dataset = data_loader.RetinexDataset(
        input_root=config.input_root,
        gt_root=config.gt_root,
        size=config.image_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 损失函数 + 优化器
    multi_scale_loss = MultiScaleConsistencyLoss().cuda()
    optimizer = optim.Adam(
        retinex_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 指标记录
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = 0
    epoch_metrics = []

    # 训练循环
    retinex_net.train()
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        for iteration, (input_img, gt_img) in enumerate(train_loader):
            input_img = input_img.cuda()
            gt_img = gt_img.cuda()

            # 核心修复：解包所有7个返回值
            enhanced, L_init, R_init, L_final, R_final, L_list, R_list = retinex_net(input_img)

            # 计算损失（逻辑不变）
            loss = multi_scale_loss(L_list, R_list, gt_img)
            epoch_loss += loss.item()

            # 反向传播（逻辑不变）
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinex_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 计算PSNR/SSIM（逻辑不变）
            output_np = enhanced.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255
            gt_np = gt_img.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255
            for b in range(output_np.shape[0]):
                epoch_psnr += calculate_psnr(output_np[b], gt_np[b])
                epoch_ssim += calculate_ssim(output_np[b], gt_np[b])

            # 打印迭代信息（逻辑不变）
            if (iteration + 1) % config.display_iter == 0:
                avg_iter_loss = epoch_loss / (iteration + 1)
                print(
                    f"Phase2 - Epoch [{epoch + 1}/{config.num_epochs}], Iter [{iteration + 1}/{len(train_loader)}], "
                    f"Batch Loss: {loss.item():.6f}, Avg Loss: {avg_iter_loss:.6f}"
                )

        # 计算epoch平均指标
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_psnr = epoch_psnr / (len(train_loader) * config.train_batch_size)
        avg_epoch_ssim = epoch_ssim / (len(train_loader) * config.train_batch_size)
        epoch_time = time.time() - epoch_start_time

        # 记录指标
        epoch_metrics.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'psnr': avg_epoch_psnr,
            'ssim': avg_epoch_ssim
        })

        # 更新最优模型
        if avg_epoch_psnr > best_psnr:
            best_psnr = avg_epoch_psnr
            best_ssim = avg_epoch_ssim
            best_epoch = epoch + 1
            save_model(retinex_net, os.path.join(snapshot_dir, f"phase2_best_model_{timestamp}.pth"), epoch + 1)

        # 打印epoch信息
        print("=" * 60)
        print(f"Phase2 - Epoch {epoch + 1} Finished | Time: {epoch_time:.2f}s")
        print(
            f"Epoch Metrics - Avg Loss: {avg_epoch_loss:.6f}, Avg PSNR: {avg_epoch_psnr:.2f} dB, Avg SSIM: {avg_epoch_ssim:.4f}")
        print(f"Current Best - Epoch {best_epoch}: PSNR {best_psnr:.2f} dB, SSIM {best_ssim:.4f}")
        print("=" * 60)

        # 保存快照
        if (epoch + 1) % config.snapshot_iter == 0:
            snapshot_path = os.path.join(snapshot_dir, f"Phase2_Epoch_{epoch + 1}_{timestamp}.pth")
            save_model(retinex_net, snapshot_path, epoch + 1)

    # 训练结束更新日志
    with open(log_file, 'a') as f:
        f.write(f"\n===== 第二阶段训练结束 =====\n")
        f.write(f"Training End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs Trained: {config.num_epochs}\n")
        f.write(f"Best Epoch: {best_epoch} | Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}\n")
        if len(epoch_metrics) >= 10:
            last_10_metrics = epoch_metrics[-10:]
            avg_last10_loss = np.mean([m['loss'] for m in last_10_metrics])
            avg_last10_psnr = np.mean([m['psnr'] for m in last_10_metrics])
            avg_last10_ssim = np.mean([m['ssim'] for m in last_10_metrics])
            f.write(
                f"Last 10 Epochs Avg - Loss: {avg_last10_loss:.6f}, PSNR: {avg_last10_psnr:.2f} dB, SSIM: {avg_last10_ssim:.4f}\n")
        f.write("=" * 70 + "\n")

    # 打印总结
    print("\n" + "=" * 70)
    print("Phase2 Training Complete - Summary Metrics")
    print("=" * 70)
    print(f"Total Epochs Trained: {config.num_epochs}")
    print(f"Best Epoch: {best_epoch} | Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    if len(epoch_metrics) >= 10:
        last_10_metrics = epoch_metrics[-10:]
        avg_last10_loss = np.mean([m['loss'] for m in last_10_metrics])
        avg_last10_psnr = np.mean([m['psnr'] for m in last_10_metrics])
        avg_last10_ssim = np.mean([m['ssim'] for m in last_10_metrics])
        print(
            f"Last 10 Epochs Avg - Loss: {avg_last10_loss:.6f}, PSNR: {avg_last10_psnr:.2f} dB, SSIM: {avg_last10_ssim:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ========== 核心修复：正确定义phase1_ckpt参数（带--的可选参数） ==========
    parser.add_argument('--phase1_ckpt', type=str, required=True,
                        help="第一阶段训练的ICRR模块权重路径",)

    # 原有数据参数
    parser.add_argument('--input_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/low")
    parser.add_argument('--gt_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/high")
    parser.add_argument('--image_size', type=int, default=512)

    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)  # Windows下设为0避免多进程报错

    # 日志与快照参数
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="")

    config = parser.parse_args()

    # 启动训练
    train(config)