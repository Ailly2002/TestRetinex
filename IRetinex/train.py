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
import datetime  # 新增：用于获取当前时间


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    # 获取当前时间作为文件夹名
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join("snapshot", timestamp)
    os.makedirs(snapshot_dir, exist_ok=True)

    # 创建记录文件
    log_file = os.path.join(snapshot_dir, f"training_log_{timestamp}.txt")

    # 写入基础信息
    with open(log_file, 'w') as f:
        f.write(f"Training Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    # 初始化模型（根据TestRetinex实际模型类调整）
    retinex_net = IRetinex().cuda()
    retinex_net.apply(weights_init)

    # 加载预训练模型（可选）
    start_epoch = 0
    if config.load_pretrain:
        if os.path.exists(config.pretrain_dir):
            retinex_net, start_epoch = load_model(retinex_net, config.pretrain_dir)
            print(f"Loaded pretrained model from {config.pretrain_dir}, start epoch: {start_epoch + 1}")
        else:
            print(f"Warning: Pretrain file not found - {config.pretrain_dir}")

    # 加载数据集（传入根目录，按相对路径匹配图片对）
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

    # 损失函数（可根据需求替换为L1/SSIM等）
    # criterion = MSELoss().cuda()
    multi_scale_loss = MultiScaleConsistencyLoss().cuda()

    # 优化器实例化
    optimizer = optim.Adam(
        retinex_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 记录全程最优指标
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = 0
    # 记录每轮指标，用于最终汇总
    epoch_metrics = []

    # 训练循环
    retinex_net.train()
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        batch_count = 0

        for iteration, (input_img, gt_img) in enumerate(train_loader):
            batch_count += 1
            # 数据移至GPU
            input_img = input_img.cuda()
            gt_img = gt_img.cuda()

            # 前向传播
            enhanced, L_list, R_list = retinex_net(input_img)
            # output_img = retinex_net(input_img)  # 模型输出增强图


            # 计算损失（增强图 vs GT图）
            # loss = criterion(output_img, gt_img)
            loss = multi_scale_loss(L_list, R_list, gt_img)

            epoch_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()

            #debug设置1 在执行 backward 之前，记录部分参数的值（用一个简单标量）
            param_sum_before = 0.0
            for p in retinex_net.parameters():
                if p.requires_grad:
                    param_sum_before += p.data.float().sum().item()
            # 计算并打印当前 loss
            # print(f"        [DEBUG] Before backward: loss={loss.item():.6f}, lr={optimizer.param_groups[0]['lr']:.6e}")

            loss.backward()

            #debug设置2 打印部分梯度统计（第一个有梯度的参数）
            # grad_norms = []
            # for i, p in enumerate(retinex_net.parameters()):
            #     if p.grad is not None:
            #         grad_norms.append(p.grad.detach().norm().item())
            # # 打印梯度的最大/平均/第一个值，若全部为 0 则说明没有梯度流入
            # if len(grad_norms) > 0:
            #     print(
            #         f"      [DEBUG] Gradients: max={max(grad_norms):.6e}, mean={(sum(grad_norms) / len(grad_norms)):.6e}, sample={grad_norms[0]:.6e}")
            # else:
            #     print("[DEBUG] No gradients found on model parameters!")

            torch.nn.utils.clip_grad_norm_(retinex_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            #debug设置3 记录参数更新后的和，比较变化量
            # param_sum_after = 0.0
            # for p in retinex_net.parameters():
            #     if p.requires_grad:
            #         param_sum_after += p.data.float().sum().item()
            #
            # print(
            #     f"[DEBUG] Param sum before={param_sum_before:.6e}, after={param_sum_after:.6e}, diff={(param_sum_after - param_sum_before):.6e}")

            # 计算当前batch的PSNR和SSIM
            # 1. 将tensor转为0-255的numpy数组（RGB格式，HWC）
            output_np = enhanced.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255
            gt_np = gt_img.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255

            # 2. 遍历batch内每张图片计算指标
            for b in range(output_np.shape[0]):
                batch_psnr = calculate_psnr(output_np[b], gt_np[b])
                batch_ssim = calculate_ssim(output_np[b], gt_np[b])
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

            # 打印迭代信息（包含当前batch损失）
            if (iteration + 1) % config.display_iter == 0:
                avg_iter_loss = epoch_loss / (iteration + 1)
                # print(
                #     # f"Epoch [{epoch + 1}/{config.num_epochs}], Iter [{iteration + 1}/{len(train_loader)}], Batch Loss: {loss.item():.6f}, Avg Loss: {avg_iter_loss:.6f}")
                #     f"Epoch [{epoch + 1}/{config.num_epochs}], Iter [{iteration + 1}/{len(train_loader)}]")
        # 计算当前epoch的平均指标
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_psnr = epoch_psnr / (len(train_loader) * config.train_batch_size)
        avg_epoch_ssim = epoch_ssim / (len(train_loader) * config.train_batch_size)
        epoch_time = time.time() - epoch_start_time

        # 记录当前epoch指标
        epoch_metrics.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'psnr': avg_epoch_psnr,
            'ssim': avg_epoch_ssim
        })

        # 更新最优指标
        if avg_epoch_psnr > best_psnr:
            best_psnr = avg_epoch_psnr
            best_ssim = avg_epoch_ssim
            best_epoch = epoch + 1
            # 保存最优模型
            save_model(retinex_net, os.path.join(snapshot_dir, f"best_model_{timestamp}.pth"), epoch + 1)

        # 打印当前epoch完整指标
        print("=" * 50)
        print(f"Epoch {epoch + 1} Finished | Time: {epoch_time:.2f}s")
        print(
            f"Epoch Metrics - Avg Loss: {avg_epoch_loss:.6f}, Avg PSNR: {avg_epoch_psnr:.2f} dB, Avg SSIM: {avg_epoch_ssim:.4f}")
        print(f"Current Best - Epoch {best_epoch}: PSNR {best_psnr:.2f} dB, SSIM {best_ssim:.4f}")
        print("=" * 50)

        # 保存模型快照
        if (epoch + 1) % config.snapshot_iter == 0:
            snapshot_path = os.path.join(snapshot_dir, f"Epoch_{epoch + 1}_{timestamp}.pth")
            save_model(retinex_net, snapshot_path, epoch + 1)

    # 训练结束，更新日志文件
    with open(log_file, 'a') as f:
        f.write(f"\nTraining End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs Trained: {config.num_epochs}\n")
        f.write(f"Best Epoch: {best_epoch} | Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}\n")
        # 输出最后10轮平均指标
        if len(epoch_metrics) >= 10:
            last_10_metrics = epoch_metrics[-10:]
            avg_last10_loss = np.mean([m['loss'] for m in last_10_metrics])
            avg_last10_psnr = np.mean([m['psnr'] for m in last_10_metrics])
            avg_last10_ssim = np.mean([m['ssim'] for m in last_10_metrics])
            f.write(
                f"Last 10 Epochs Avg - Loss: {avg_last10_loss:.6f}, PSNR: {avg_last10_psnr:.2f} dB, SSIM: {avg_last10_ssim:.4f}\n")
        f.write("=" * 60 + "\n")

    # 打印总结信息
    print("\n" + "=" * 60)
    print("Training Complete - Summary Metrics")
    print("=" * 60)
    print(f"Total Epochs Trained: {config.num_epochs}")
    print(f"Best Epoch: {best_epoch} | Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    if len(epoch_metrics) >= 10:
        last_10_metrics = epoch_metrics[-10:]
        avg_last10_loss = np.mean([m['loss'] for m in last_10_metrics])
        avg_last10_psnr = np.mean([m['psnr'] for m in last_10_metrics])
        avg_last10_ssim = np.mean([m['ssim'] for m in last_10_metrics])
        print(
            f"Last 10 Epochs Avg - Loss: {avg_last10_loss:.6f}, PSNR: {avg_last10_psnr:.2f} dB, SSIM: {avg_last10_ssim:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据目录参数
    parser.add_argument('--input_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/low")  # Input根目录
    parser.add_argument('--gt_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/high")  # GT根目录
    parser.add_argument('--image_size', type=int, default=256)

    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # 日志与快照参数
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Retinex/")  # 这个参数现在不再使用
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots_Retinex/Epoch_99.pth")

    config = parser.parse_args()

    # 启动训练
    train(config)