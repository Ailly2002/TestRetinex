import torch
import torch.nn as nn

# ======================
# 1. Dual Color Space Prior Module
# ======================
class DualColorSpacePrior(nn.Module):
    """双色空间先验模块 (严格遵循论文公式(9)和(10))"""
    def __init__(self):
        super(DualColorSpacePrior, self).__init__()
        # 5通道输入 -> 3通道输出
        self.conv = nn.Sequential(
            nn.Conv2d(5, 3, 1, 1, 0),  # 1x1卷积
            nn.Conv2d(3, 3, 5, 1, 2),  # 5x5卷积
            nn.Conv2d(3, 3, 1, 1, 0)   # 1x1卷积
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, 3, H, W] 低光照图像
        输出: [B, 3, H, W] 照明图估计
        """
        # 1. 计算RGB空间照明先验 (公式(10))
        Lrgb = (x[:, 0:1] + x[:, 1:2] + x[:, 2:3]) / 3  # 1通道
        
        # 2. 直接计算V通道 (亮度) 作为LHSV
        max_val, _ = torch.max(x, dim=1, keepdim=True)  # 1通道
        Lhsv = max_val
        
        # 3. 拼接输入特征、RGB照明先验和V通道,应用卷积得到最终照明估计
        concat_features = torch.cat([x, Lrgb, Lhsv], dim=1)  # 5通道
        Linit = self.conv(concat_features)
        return Linit

# ======================
# 2. Reflectance Decomposition
# ======================
class ReflectanceDecomposition(nn.Module):
    """反射率分解模块 (严格遵循论文3.3节公式(10))"""
    def __init__(self):
        super(ReflectanceDecomposition, self).__init__()
        # 4通道输入 -> 3通道输出
        self.conv = nn.Sequential(
            nn.Conv2d(4, 3, 1, 1, 0),  # 1x1卷积
            nn.Conv2d(3, 3, 5, 1, 2),  # 5x5卷积
            nn.Conv2d(3, 3, 1, 1, 0)   # 1x1卷积
        )
    
    def forward(self, I_l: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        输入:
            I_l: [B, 3, H, W] 低光照图像
            L:   [B, 3, H, W] 照明估计
        输出:
            R:   [B, 3, H, W] 反射率图 (R = I / L)
        """
        softmax_part = F.softmax(I_l / (L + 1e-8), dim=1)

        concat_features = torch.cat([I_l, softmax_part], dim=1)  # 4通道
        Rinit = self.conv(concat_features)
        return Rinit