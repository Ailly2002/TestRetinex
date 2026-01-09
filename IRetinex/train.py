import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from models import IRetinex, MultiScaleConsistencyLoss
import data_loader
from utils import save_model, visualize_results, calculate_psnr, calculate_ssim, load_model
import datetime
import torch.nn.functional as F
import math  # 新增：用于计算累积步数


"""
说明：
- 函数 plot_training_metrics 接受 epoch_metrics（list of dict）和 snapshot_dir 路径。
- 画出 Loss / PSNR / SSIM 随 epoch 的曲线，PSNR vs SSIM 散点图，以及 PSNR 改进的直方图（基于相邻 epoch 差分）。
- 调用示例：在训练结束后执行 plot_training_metrics(epoch_metrics, snapshot_dir)
"""

def plot_training_metrics(epoch_metrics, snapshot_dir):
    os.makedirs(snapshot_dir, exist_ok=True)

    epochs = [m['epoch'] for m in epoch_metrics]
    losses = [m['loss'] for m in epoch_metrics]
    psnrs = [m['psnr'] for m in epoch_metrics]
    ssims = [m['ssim'] for m in epoch_metrics]

    # Loss / PSNR / SSIM over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, '-o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnrs, '-o', label='PSNR (dB)')
    plt.plot(epochs, ssims, '-o', label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, 'psnr_ssim_curve.png'))
    plt.close()

    # PSNR vs SSIM scatter (epoch-wise averages)
    plt.figure(figsize=(6, 6))
    plt.scatter(psnrs, ssims, c=epochs, cmap='viridis', s=40)
    for x, y, e in zip(psnrs, ssims, epochs):
        if (e == epochs[0]) or (e == epochs[-1]) or (e % max(1, len(epochs)//5) == 0):
            plt.text(x, y, str(e), fontsize=8)
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.colorbar(label='Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, 'psnr_vs_ssim.png'))
    plt.close()

    # PSNR improvements histogram (epoch-to-epoch delta)
    if len(psnrs) >= 2:
        deltas = np.diff(psnrs)
        plt.figure(figsize=(8, 5))
        plt.hist(deltas, bins=20, color='C1', edgecolor='k', alpha=0.8)
        plt.xlabel('Delta PSNR (dB) between successive epochs')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(snapshot_dir, 'psnr_delta_hist.png'))
        plt.close()

    # 可选：保存数值到 csv 以便后续分析
    # import csv
    # csv_path = os.path.join(snapshot_dir, 'epoch_metrics.csv')
    # with open(csv_path, 'w', newline='') as cf:
    #     writer = csv.DictWriter(cf, fieldnames=['epoch', 'loss', 'psnr', 'ssim'])
    #     writer.writeheader()
    #     for m in epoch_metrics:
    #         writer.writerow({'epoch': m['epoch'], 'loss': m['loss'], 'psnr': m['psnr'], 'ssim': m['ssim']})


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

    # 新增：计算梯度累积步数，使等效 batch_size >= 32
    accumulation_steps = max(1, math.ceil(16.0 / float(config.train_batch_size)))
    # 将累积信息写入日志
    with open(log_file, 'a') as f:
        f.write(f"Gradient Accumulation Steps: {accumulation_steps} (effective batch_size ~= {config.train_batch_size * accumulation_steps})\n")

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

        # 在 epoch 开始时清零优化器梯度，之后按 accumulation_steps 做累计
        optimizer.zero_grad()

        for iteration, (input_img, gt_img) in enumerate(train_loader):
            batch_count += 1
            # 数据移至GPU
            input_img = input_img.cuda()
            gt_img = gt_img.cuda()

            # 前向传播
            outputs = retinex_net(input_img)

            # 解析模型输出（支持 tensor / tuple(list) / dict）
            enhanced = None
            L_list = None
            R_list = None

            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 0:
                    raise RuntimeError("Model returned empty tuple/list")
                # 支持两种情况：
                # 1) 老模型只返回 enhanced
                # 2) 新模型返回 (enhanced, L_init, R_init, L_final, R_final, L_list, R_list)
                enhanced = outputs[0]
                if len(outputs) >= 7:
                    # 新模型约定：索引5/6 为后5个RCM特征列表
                    # outputs[5] 和 outputs[6] 应为 list/tuple 且长度为5
                    L_list = outputs[5]
                    R_list = outputs[6]
                elif len(outputs) >= 3:
                    # 兼容老格式：第二项可能是 L_list（如果是列表）
                    candidate1 = outputs[1]
                    candidate2 = outputs[2] if len(outputs) > 2 else None
                    if isinstance(candidate1, (list, tuple)) and isinstance(candidate2, (list, tuple)):
                        L_list = candidate1
                        R_list = candidate2
                    else:
                        # 可能为 (enhanced, dict) 或 (enhanced, tensor...)
                        if len(outputs) == 2 and isinstance(outputs[1], dict):
                            d = outputs[1]
                            L_list = d.get('L_list') or d.get('L')
                            R_list = d.get('R_list') or d.get('R')
                        else:
                            # 保持 L_list/R_list 为 None，后续会尝试合成尺度或回退
                            L_list = None
                            R_list = None
            elif isinstance(outputs, dict):
                enhanced = outputs.get('enhanced') or outputs.get('output') or outputs.get('pred') or outputs.get(
                    'enhance')
                L_list = outputs.get('L_list') or outputs.get('L') or outputs.get('illum')
                R_list = outputs.get('R_list') or outputs.get('R') or outputs.get('refl')

            else:
                enhanced = outputs

            if isinstance(enhanced, (tuple, list)):
                enhanced = enhanced[0]

            # 如果已有合法的 5 尺度列表，直接使用 multi_scale_loss
            use_multi_scale = (
                    isinstance(L_list, (list, tuple)) and isinstance(R_list, (list, tuple))
                    and len(L_list) == 5 and len(R_list) == 5
            )

            if use_multi_scale:
                loss = multi_scale_loss(L_list, R_list, gt_img)
            else:
                # 尝试从 enhanced 合成 5 个尺度用于 multi_scale_loss（较稳妥的退路）
                try:
                    if enhanced is None:
                        raise ValueError("No enhanced output available to synthesize scales")

                    # 确保 enhanced 是 float tensor，并与 gt 同设备
                    enhanced = enhanced.to(gt_img.device).float()

                    # 合成 5 个尺度：scale 0 是原分辨率，之后逐次下采样 0.5
                    L_synth = []
                    R_synth = []
                    curr = enhanced
                    for i in range(5):
                        L_synth.append(curr)
                        R_synth.append(curr)
                        if i < 4:
                            # 使用双线性插值下采样因子 0.5
                            h = max(1, int(curr.shape[2] // 2))
                            w = max(1, int(curr.shape[3] // 2))
                            curr = F.interpolate(curr, size=(h, w), mode='bilinear', align_corners=False)

                    # 验证合成列表长度
                    if len(L_synth) == 5 and len(R_synth) == 5:
                        loss = multi_scale_loss(L_synth, R_synth, gt_img)
                    else:
                        raise RuntimeError("Synthesized scales count mismatch")

                except Exception:
                    # 最后退回像素级 MSE 损失，保证训练不中断
                    criterion_fallback = MSELoss().to(gt_img.device)
                    if enhanced is None:
                        # 如果连 enhanced 都没有，强制计算模型一次得到增强图再计算 MSE
                        enhanced = retinex_net(input_img)
                        if isinstance(enhanced, (tuple, list)):
                            enhanced = enhanced[0]
                    loss = criterion_fallback(enhanced, gt_img)


            # 记录原始 loss 数值用于统计（未缩放）
            loss_value = float(loss.item())
            epoch_loss += loss_value

            # 梯度累积：缩放 loss 后 backward
            scaled_loss = loss / accumulation_steps

            # debug：记录参数和其它信息（在 backward 前）
            param_sum_before = 0.0
            for p in retinex_net.parameters():
                if p.requires_grad:
                    param_sum_before += p.data.float().sum().item()

            # 检查 loss 是否有限
            if not math.isfinite(loss_value):
                err_info = f"Non-finite loss at epoch={epoch + 1}, iter={iteration + 1}, loss={loss_value}"
                print(err_info)
                bad_dir = os.path.join(snapshot_dir, "bad_batches")
                os.makedirs(bad_dir, exist_ok=True)
                try:
                    torch.save({
                        'input': input_img.detach().cpu(),
                        'gt': gt_img.detach().cpu(),
                        'epoch': epoch + 1,
                        'iteration': iteration + 1,
                        'loss': float(loss_value)
                    }, os.path.join(bad_dir, f"bad_epoch{epoch + 1}_iter{iteration + 1}.pt"))
                except Exception as e:
                    print("Failed to save bad batch:", e)
                raise RuntimeError(err_info)

            # 反向传播（累积）
            scaled_loss.backward()

            # 何时执行一次 optimizer.step(): 当累计到 accumulation_steps 或到达最后一个 batch
            is_last_step = ((iteration + 1) % accumulation_steps == 0) or ((iteration + 1) == len(train_loader))
            if is_last_step:
                # 梯度裁剪并更新参数
                torch.nn.utils.clip_grad_norm_(retinex_net.parameters(), config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            # 计算当前batch的PSNR和SSIM（不受梯度累积影响）
            output_np = enhanced.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255
            gt_np = gt_img.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255

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

    plot_training_metrics(epoch_metrics, snapshot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据目录参数
    parser.add_argument('--input_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/low")  # Input根目录
    parser.add_argument('--gt_root', type=str, default="E:/Low-LightDatasets/Images/LOLdataset/our485/high")  # GT根目录
    parser.add_argument('--image_size', type=int, default=512)

    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    # 日志与快照参数
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Retinex/")  # 这个参数现在不再使用
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots_Retinex/Epoch_99.pth")

    config = parser.parse_args()
    train(config)