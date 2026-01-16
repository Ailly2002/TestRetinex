import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 1. Dual Color Space Prior Module
# ======================
class DualColorSpacePrior(nn.Module):
    """双色空间先验模块 (严格遵循论文公式(9)和(10))"""
    def __init__(self):
        super(DualColorSpacePrior, self).__init__()
        input_channels = 5  # 3 (I_l) + 1 (Lrgb) + 1 (Lhsv)
        mid_channels = 40
        # 1x1 -> 5x5 depthwise -> 1x1 -> ReLU
        self.prior_net = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, stride=1, padding=1,groups=1, bias=True, padding_mode='replicate'),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, groups=mid_channels, bias=True, padding_mode='replicate'),  # depthwise
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, 3, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='replicate'),
            # nn.BatchNorm2d(3),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, 3, H, W] 低光照图像
        输出: [B, 3, H, W] 照明图估计
        """
        # 1. 计算RGB空间照明先验 (公式(10))
        Lrgb = (x[:, 0:1] + x[:, 1:2] + x[:, 2:3]) / 3  # 1通道

        # 2. 直接计算V通道 (明度) 作为LHSV
        Lhsv = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W], 1通道

        # 3. 拼接输入特征、RGB照明先验和V通道,应用卷积得到最终照明估计
        concat_features = torch.cat([x, Lrgb, Lhsv], dim=1)  # 5通道

        Linit = self.prior_net(concat_features)
        return Linit

# ======================
# 2. Reflectance Decomposition
# ======================
class ReflectanceDecomposition(nn.Module):
    """反射率分解模块 (严格遵循论文3.3节公式(10))"""
    def __init__(self):
        super(ReflectanceDecomposition, self).__init__()
        input_channels = 6  # 3 (I_l) + 3 (softmax部分)
        mid_channels = 40
        # 使用 nn.Sequential: 1x1 -> BN -> ReLU -> 5x5 depthwise -> BN -> ReLU -> 1x1 -> BN -> ReLU
        self.reflect_net = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, groups=mid_channels, bias=True),  # depthwise
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, 3, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(3),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True)
        )

    def forward(self, I_l: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        输入:
            I_l: [B, 3, H, W] 低光照图像
            L:   [B, 3, H, W] 照明估计 (R = I / L)
        输出:
            R:   [B, 3, H, W] 反射率图
        """
        reflectance_prior = I_l / (L + 1e-8)  # [B, 3, H, W]

        # 在空间维度(H*W)上应用softmax，保持每个通道独立
        B, C, H, W = reflectance_prior.shape
        reflectance_flat = reflectance_prior.view(B, C, -1)  # [B, 3, H*W]
        softmax_spatial = F.softmax(reflectance_flat, dim=-1)  # 在H*W维度做softmax
        softmax_part = softmax_spatial.view(B, C, H, W)  # [B, 3, H, W]

        concat_features = torch.cat([I_l, softmax_part], dim=1)  # 6通道

        Rinit = self.reflect_net(concat_features)
        return Rinit


# ======================
# 3. Inter-component Residual Reduction (ICRR)
# 实际未使用
# ======================
class ICRR(nn.Module):
    """互成分残差缩减模块 (ICRR) - 严格遵循论文3.3节"""

    def __init__(self):
        super(ICRR, self).__init__()
        self.dual_color_prior = DualColorSpacePrior()
        self.reflectance_decomp = ReflectanceDecomposition()

    def forward(self, I_l: torch.Tensor) -> tuple:
        """
        输入:
            I_l: [B, 3, H, W] 低光照图像
        输出:
            L_init: [B, 3, H, W] 照明估计
            R_init: [B, 3, H, W] 反射率估计
        """
        # 1. 调用DualColorSpacePrior获得Linit
        L_init = self.dual_color_prior(I_l)

        # 2. 将Linit作为输入，从ReflectanceDecomposition获得Rinit
        R_init = self.reflectance_decomp(I_l, L_init)

        return L_init, R_init