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
    """前馈网络 (FFN) - 论文3.4节"""
    def __init__(self, channels: int):
        super(FFN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        return x

class RCM(nn.Module):
    """残差缓解与组件增强模块 (RCM) - 论文3.4节"""
    def __init__(self, channels: int):
        super(RCM, self).__init__()
        self.ses = SES(channels)
        self.mres_l = MRES(channels)
        self.mres_r = MRES(channels)
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
    """完整模型 (论文3.3-3.5节)，支持5尺度RCM应用"""
    def __init__(self, feature_channels=64):
        super(IRetinex, self).__init__()
        self.dual_color = DualColorSpacePrior()
        self.reflectance_decomp = ReflectanceDecomposition()

        # 特征提取器：将3通道转换为高维特征（64通道）
        self.feature_extractor_l = FeatureExtractor(3, feature_channels)
        self.feature_extractor_r = FeatureExtractor(3, feature_channels)

        # RCM模块使用正确的通道数（64）
        self.rcm = RCM(feature_channels)

        # 特征重建器：将高维特征转换回3通道
        self.feature_reconstructor_l = FeatureReconstructor(feature_channels, 3)
        self.feature_reconstructor_r = FeatureReconstructor(feature_channels, 3)

    def forward(self, x: torch.Tensor) -> tuple:
        # ICRR模块：初始化L和R (3通道)
        L_init = self.dual_color(x)
        R_init = self.reflectance_decomp(x, L_init)

        # 特征提取：将3通道转换为高维特征（64通道）
        L_feat = self.feature_extractor_l(L_init)
        R_feat = self.feature_extractor_r(R_init)

        # 保存初始特征用于返回
        L_feat_list = [L_feat]
        R_feat_list = [R_feat]

        # 应用5次RCM（在高维特征空间中）
        for _ in range(5):
            L_enhanced, R_enhanced = self.rcm(L_feat_list[-1], R_feat_list[-1])
            L_feat_list.append(L_enhanced)
            R_feat_list.append(R_enhanced)

        # 重建最终的L和R（从高维特征回到3通道）
        L_final = self.feature_reconstructor_l(L_feat_list[-1])
        R_final = self.feature_reconstructor_r(R_feat_list[-1])

        # 最终增强图像
        enhanced = L_final * R_final

        # 重建中间结果（用于损失计算）
        # L_list = [self.feature_reconstructor_l(feat) for feat in L_feat_list[:-1]]
        # R_list = [self.feature_reconstructor_r(feat) for feat in R_feat_list[:-1]]
        L_list = L_feat_list[:-1]  # 5个尺度的64通道特征 (batch, 64, H, W)
        R_list = R_feat_list[:-1]  # 5个尺度的64通道特征 (batch, 64, H, W)

        return enhanced, L_list, R_list