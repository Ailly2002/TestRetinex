import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConsistencyLoss(nn.Module):
    """多尺度一致性损失：适配9个RCM的U型网络，使用后5个RCM输出计算"""

    def __init__(self):
        super(MultiScaleConsistencyLoss, self).__init__()
        # 卷积层：64通道→3通道（匹配RCM输出的特征维度）
        self.conv_r = nn.ModuleList([nn.Conv2d(64, 3, 3, 1, 1) for _ in range(5)])
        self.conv_l = nn.ModuleList([nn.Conv2d(64, 3, 3, 1, 1) for _ in range(5)])

        # 上采样器：每次上采样2倍，用于恢复到原始尺度
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 3, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(5)
        ])

    def forward(self, L_list: list, R_list: list, target: torch.Tensor) -> torch.Tensor:
        # 确保输入是后5个RCM的输出
        assert len(L_list) == 5 and len(R_list) == 5, "输入必须包含5个RCM输出特征"
        # loss = 0.0
        loss = torch.tensor(0.0, device=target.device)
        target_h, target_w = target.shape[2:]  # 原始目标图像尺寸

        for j in range(5):
            # 1. 将64通道特征转换为3通道
            Ij_r = self.conv_r[j](R_list[j])
            Ij_l = self.conv_l[j](L_list[j])
            Ij = Ij_r * Ij_l  # 论文中的Ij计算逻辑

            # 2. 动态上采样到target尺寸：后5个RCM的尺度依次为 1/32 → 1/16 → 1/8 → 1/4 → 1/2
            #    对应需要上采样 5-j 次（j=0需5次，j=4需1次）
            Ij_up = Ij
            for _ in range(5 - j):
                Ij_up = self.upsamplers[_](Ij_up)

            # 3. 强制对齐尺寸（防止上采样后尺寸偏差）
            Ij_up = F.interpolate(Ij_up, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

            # 4. 累加L1损失
            loss += F.l1_loss(Ij_up, target)

        # 损失平均
        loss = loss / 5

        return loss