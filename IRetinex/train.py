import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import IRetinex, MultiScaleConsistencyLoss
from data_loader import get_dataloader
from utils import save_model, visualize_results

def train():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = IRetinex().to(device)
    loss_fn = MultiScaleConsistencyLoss().to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 数据加载
    train_loader = get_dataloader('data/train', batch_size=4)
    
    # TensorBoard
    writer = SummaryWriter('runs/iretinex')
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            # 前向传播
            enhanced, L_list, R_list = model(images)
            
            # 计算损失
            loss = loss_fn(L_list, R_list, images)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每10个batch打印一次
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 每个epoch保存模型
        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')
        
        # 保存模型
        save_path = f'checkpoints/iretinex_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        save_model(model, save_path, epoch+1)
        
        # TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # 验证集可视化
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(train_loader))
                val_batch = val_batch.to(device)
                enhanced, _, _ = model(val_batch)
                visualize_results(val_batch[0], enhanced[0], val_batch[0], 
                                 f'vis/epoch_{epoch+1}.png')
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    train()