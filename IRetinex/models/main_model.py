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
        similarity = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.di
        similarity = F.softmax(similarity, dim=-1)
        residual = torch.matmul(similarity, V.permute(0, 1, 3, 2))
        return residual.permute(0, 1, 3, 2)

class SES(nn.Module):
    """超分辨率增强方案 (SES) - 论文3.4节"""
    def __init__(self, channels: int, scale_factor: int = 2):
        super(SES, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * (scale_factor ** 2), 
                             kernel_size=3, stride=scale_factor, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)

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
        x = self.gelu(self.conv3(x))
        return x

class RCM(nn.Module):
    """残差缓解与组件增强模块 (RCM) - 论文3.4节"""
    def __init__(self, channels: int):
        super(RCM, self).__init__()
        self.ses = SES(channels)
        self.mres_l = MRES(channels)
        self.mres_r = MRES(channels)
        self.ffn = FFN(channels)
    
    def forward(self, L: torch.Tensor, R: torch.Tensor) -> tuple:
        L_super = self.ses(L)
        R_super = self.ses(R)
        
        Qr = self.mres_l.q_layer(R_super)
        Kl = self.mres_l.k_layer(L_super)
        Vr = self.mres_l.v_layer(R)
        residual_L = self.mres_l(Qr, Kl, Vr)
        
        Ql = self.mres_r.q_layer(L_super)
        Kr = self.mres_r.k_layer(R_super)
        Vl = self.mres_r.v_layer(L)
        residual_R = self.mres_r(Ql, Kr, Vl)
        
        L_enhanced = self.ffn(residual_L + L)
        R_enhanced = self.ffn(residual_R + R)
        
        L_enhanced = F.layer_norm(L_enhanced, L_enhanced.shape[1:])
        R_enhanced = F.layer_norm(R_enhanced, R_enhanced.shape[1:])
        
        return L_enhanced, R_enhanced

class IRetinex(nn.Module):
    """完整模型 (论文3.3-3.5节)，支持5尺度RCM应用"""
    def __init__(self):
        super(IRetinex, self).__init__()
        self.dual_color = DualColorSpacePrior()
        self.reflectance_decomp = ReflectanceDecomposition()
        self.rcm = RCM(64)
    
    def forward(self, x: torch.Tensor) -> tuple:
        L_init = self.dual_color(x)
        R_init = self.reflectance_decomp(x, L_init)
        
        L_list = [L_init]
        R_list = [R_init]
        
        for _ in range(5):
            L_enhanced, R_enhanced = self.rcm(L_list[-1], R_list[-1])
            L_list.append(L_enhanced)
            R_list.append(R_enhanced)
        
        enhanced = L_list[-1] * R_list[-1]
        return enhanced, L_list[:5], R_list[:5]