import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConsistencyLoss(nn.Module):
    """多尺度一致性损失 (论文3.5节)"""
    def __init__(self):
        super(MultiScaleConsistencyLoss, self).__init__()
        self.conv_r = nn.ModuleList([nn.Conv2d(64, 3, 3, 1, 1) for _ in range(5)])
        self.conv_l = nn.ModuleList([nn.Conv2d(64, 3, 3, 1, 1) for _ in range(5)])
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 3, 3, 1, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(5)
        ])

    def forward(self, L_list: list, R_list: list, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for j in range(5):
            Ij_r = self.conv_r[j](R_list[j])
            Ij_l = self.conv_l[j](L_list[j])
            Ij = Ij_r * Ij_l

            Ij_up = Ij
            for _ in range(4 - j):
                Ij_up = self.upsamplers[_](Ij_up)

            loss += F.l1_loss(Ij_up, target)

        return loss