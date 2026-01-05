import torch
import torch.nn as nn
import torch.nn.functional as F
from .icrr import DualColorSpacePrior, ReflectanceDecomposition

class MRES(nn.Module):
    """互残差估计模块 (MRES) - 论文3.4节"""
    def __init__(self, channels: int):
        super(MRES, self).__init__()
        self.q_layer = nn.Conv2d(channels, channels, 1)
        self.k_layer = nn.Conv2d(channels, channels, 1)
        self.v_layer = nn.Conv2d(channels, channels, 1)
        self.di = nn.Parameter(torch.ones(1))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Reshape for matrix multiplication: [B, C, H, W] -> [B, C, H*W]
        B, C, H, W = Q.shape
        Q_flat = Q.view(B, C, -1)  # [B, C, HW]
        K_flat = K.view(B, C, -1)  # [B, C, HW]
        V_flat = V.view(B, C, -1)  # [B, C, HW]

        # Compute similarity: [B, C, HW] @ [B, C, HW].T -> [B, C, HW]
        similarity = torch.matmul(Q_flat, K_flat.transpose(-2, -1)) * self.di
        similarity = F.softmax(similarity, dim=-1)

        # Apply attention: [B, C, HW] @ [B, C, HW] -> [B, C, HW]
        residual = torch.matmul(similarity, V_flat)
        return residual.view(B, C, H, W)


class SES(nn.Module):
    """超分辨率增强方案 (SES) - 论文3.4节"""

    def __init__(self, channels: int):
        super(SES, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enhance(x)

class FFN(nn.Module):
    """下采样版前馈网络 (FFN) - 用于前5个RCM，尺度减半"""
    def __init__(self, channels: int):
        super(FFN, self).__init__()
        # FFN层当中一层stride=2的卷积实现尺度减半
        self.conv1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        return x

class FFN_Upsample(nn.Module):
    """上采样版前馈网络 (FFN) - 用于后4个RCM，尺度加倍"""
    def __init__(self, channels: int):
        super(FFN_Upsample, self).__init__()
        # 先上采样2倍，再卷积，实现尺度加倍
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)  # 先上采样2倍
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        return x

class RCM(nn.Module):
    """残差缓解与组件增强模块 (RCM) - 支持上下采样切换"""
    def __init__(self, channels: int, is_upsample: bool = False):
        super(RCM, self).__init__()
        self.ses = SES(channels)
        self.mres_l = MRES(channels)
        self.mres_r = MRES(channels)
        # 根据是否上采样选择FFN类型
        if is_upsample:
            self.ffn_l = FFN_Upsample(channels)
            self.ffn_r = FFN_Upsample(channels)
        else:
            self.ffn_l = FFN(channels)
            self.ffn_r = FFN(channels)

    def forward(self, L: torch.Tensor, R: torch.Tensor) -> tuple:
        L_super = self.ses(L)
        R_super = self.ses(R)

        # MRES for illumination: transfer texture from reflectance to illumination
        Qr = self.mres_l.q_layer(R_super)
        Kl = self.mres_l.k_layer(L_super)
        Vr = self.mres_l.v_layer(R)
        residual_L = self.mres_l(Qr, Kl, Vr)

        # MRES for reflectance: transfer light from illumination to reflectance
        Ql = self.mres_r.q_layer(L_super)
        Kr = self.mres_r.k_layer(R_super)
        Vl = self.mres_r.v_layer(L)
        residual_R = self.mres_r(Ql, Kr, Vl)

        L_enhanced = self.ffn_l(residual_L + L)
        R_enhanced = self.ffn_r(residual_R + R)

        return L_enhanced, R_enhanced

class FeatureExtractor(nn.Module):
    """特征提取器：将3通道图像转换为高维特征"""
    def __init__(self, in_channels=3, out_channels=64):
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.extractor(x)

class FeatureReconstructor(nn.Module):
    """特征重建器：将高维特征转换回3通道图像"""
    def __init__(self, in_channels=64, out_channels=3):
        super(FeatureReconstructor, self).__init__()
        self.reconstructor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )

    def forward(self, x):
        return self.reconstructor(x)

class IRetinex(nn.Module):
    """9个RCM构成U型网络（前5下采样+后4上采样）"""

    def __init__(self, feature_channels=64):
        super(IRetinex, self).__init__()
        self.dual_color = DualColorSpacePrior()
        self.reflectance_decomp = ReflectanceDecomposition()

        # 特征提取器：3通道→64通道
        self.feature_extractor_l = FeatureExtractor(3, feature_channels)
        self.feature_extractor_r = FeatureExtractor(3, feature_channels)

        # RCM模块：前5个下采样，后4个上采样（共9个）
        self.rcm_down = nn.ModuleList([RCM(feature_channels, is_upsample=False) for _ in range(5)])
        self.rcm_up = nn.ModuleList([RCM(feature_channels, is_upsample=True) for _ in range(4)])

        # 特征重建器：64通道→3通道
        self.feature_reconstructor_l = FeatureReconstructor(feature_channels, 3)
        self.feature_reconstructor_r = FeatureReconstructor(feature_channels, 3)

        # 新增：最终上采样层（1/2尺寸→原始尺寸）
        self.final_upsample = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Sigmoid()  # 保持输出在[0,1]范围
        )

    def forward(self, x: torch.Tensor) -> tuple:
        # 1. ICRR模块初始化L和R（3通道）
        L_init = self.dual_color(x)
        R_init = self.reflectance_decomp(x, L_init)

        # 2. 特征提取：3通道→64通道高维特征
        L_feat = self.feature_extractor_l(L_init)
        R_feat = self.feature_extractor_r(R_init)

        # 3. 执行9个RCM（前5下采样+后4上采样）
        all_L_feats = []  # 存储所有9个RCM的L特征输出
        all_R_feats = []  # 存储所有9个RCM的R特征输出

        # 前5个下采样RCM
        for rcm in self.rcm_down:
            L_feat, R_feat = rcm(L_feat, R_feat)
            all_L_feats.append(L_feat)
            all_R_feats.append(R_feat)

        # 后4个上采样RCM
        for rcm in self.rcm_up:
            L_feat, R_feat = rcm(L_feat, R_feat)
            all_L_feats.append(L_feat)
            all_R_feats.append(R_feat)

        # 4. 提取后5个RCM输出（第5个下采样 + 后4个上采样）
        L_list = all_L_feats[4:]  # 索引4-8（共5个）
        R_list = all_R_feats[4:]

        # 5. 重建最终的L和R（从最后一个RCM输出恢复3通道）
        L_final = self.feature_reconstructor_l(all_L_feats[-1])
        R_final = self.feature_reconstructor_r(all_R_feats[-1])

        # 6. 初始增强图像（1/2尺寸）+ 最终上采样（恢复原始尺寸）
        enhanced_half = L_final * R_final
        enhanced = self.final_upsample(enhanced_half)  # 128×128 → 256×256

        # 调试：打印关键尺寸
        # print(f"输入图像尺寸: {x.shape}")
        # print(f"1/2尺寸增强图: {enhanced_half.shape}")
        # print(f"最终增强图尺寸: {enhanced.shape}")
        # print("后5个RCM的L特征尺寸：")
        # for i, feat in enumerate(L_list):
        #     print(f"第{i + 5}个RCM - L特征尺寸: {feat.shape}")

        return enhanced, L_list, R_list