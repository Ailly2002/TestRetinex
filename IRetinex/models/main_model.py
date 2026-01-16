import torch
import torch.nn as nn
import torch.nn.functional as F
from .icrr import DualColorSpacePrior, ReflectanceDecomposition


# class MRES(nn.Module):
#     """互残差估计模块 (MRES) - 论文3.4节"""
#     def __init__(self, channels: int):
#         super(MRES, self).__init__()
#         self.q_layer = nn.Conv2d(channels, channels, 1)
#         self.k_layer = nn.Conv2d(channels, channels, 1)
#         self.v_layer = nn.Conv2d(channels, channels, 1)
#         self.di = nn.Parameter(torch.ones(1))
#         self.layernorm = nn.LayerNorm(channels)
#
#     def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
#         # Reshape for matrix multiplication: [B, C, H, W] -> [B, C, H*W]
#         B, C, H, W = Q.shape
#         Q_flat = Q.view(B, C, -1)  # [B, C, HW]
#         K_flat = K.view(B, C, -1)  # [B, C, HW]
#         V_flat = V.view(B, C, -1)  # [B, C, HW]
#
#         # Compute similarity: [B, C, HW] @ [B, C, HW].T -> [B, C, HW]
#         similarity = torch.matmul(Q_flat, K_flat.transpose(-2, -1)) * self.di
#         similarity = F.softmax(similarity, dim=-1)
#
#         # Apply attention: [B, C, HW] @ [B, C, HW] -> [B, C, HW]
#         residual = torch.matmul(similarity, V_flat)
#
#         # 假定该 forward 的输入有 Q, K, V 三个张量（或至少含有 V），并且 residual 为前面计算得到的张量。
#
#         # --- 开始替换片段 ---
#         # residual: 之前计算得到的张量，可能是 [B, N] 或其他展平形式
#         # V: 传入的 value 特征张量，包含正确的空间尺寸
#
#         B = residual.size(0)
#
#         # 从 V 获取目标空间尺寸（更可靠）
#         H = V.size(2)
#         W = V.size(3)
#
#         # 计算每个 batch 的元素数（按展平张量）
#         num_per_batch = residual.numel() // B
#
#         # 检查能否整除 H*W
#         if (H * W) == 0:
#             raise RuntimeError(f"目标空间尺寸非法 H*W == 0 (H={H}, W={W})")
#         if num_per_batch % (H * W) != 0:
#             raise RuntimeError(
#                 f"无法 reshape residual: 每批元素数 {num_per_batch} 不能被 H*W={H * W} 整除. "
#                 f"residual.shape={tuple(residual.shape)}, V.shape={tuple(V.shape)}"
#             )
#
#         C = num_per_batch // (H * W)
#
#         # 最终 reshape（使用 view 或 reshape 都可）
#         residual = residual.view(B, C, H, W)
#
#         # 在输出前对 residual 做 LayerNorm（按每个空间位置的通道归一化）
#         # 将形状 (B, C, H, W) -> (B, H, W, C) 以便 LayerNorm(normalized_shape=channels) 在最后一个维度上生效
#         residual = residual.permute(0, 2, 3, 1)  # (B, H, W, C)
#         residual = self.layernorm(residual)
#         residual = residual.permute(0, 3, 1, 2)  # (B, C, H, W)
#
#         return residual


class MRES(nn.Module):
    """互残差估计模块 (MRES)
    - 支持 Q, K 空间尺寸一致，V 可以为不同空间尺寸（会被重采样到 Q 的尺寸）
    - 假设外部可能已经用 mres.q_layer/k_layer/v_layer 对输入做了投影，因此 forward 不再重复应用这些层
    """
    def __init__(self, channels: int, interp_mode: str = 'bilinear', eps: float = 1e-6):
        super(MRES, self).__init__()
        # 这些 1x1 层通常在 RCM 中被外部调用 (RCM 中调用 mres.q_layer/k_layer/v_layer)
        self.q_layer = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_layer = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_layer = nn.Conv2d(channels, channels, 1, bias=False)

        # 可学习缩放因子 di，用作除数（论文中为除以 d_i）
        self.di = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 若 V 的通道数和 Q 不一致时，用于投影 V 到 C
        self._v_proj = None  # 延迟创建以节省不必要参数

        self.layernorm = nn.LayerNorm(channels)
        self.interp_mode = interp_mode
        self.eps = eps

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        输入：
          Q: (B, C_q, H_q, W_q)
          K: (B, C_k, H_q, W_q)  -- 空间尺寸应与 Q 相同
          V: (B, C_v, H_v, W_v)  -- 空间尺寸可以不同
        返回：
          residual: (B, C_out, H_q, W_q)  通道数与 Q/K 的通道数一致（C_out == C_q）
        注意：通常 Q/K/V 已经是通过 mres.q_layer/k_layer/v_layer 投影后的特征（由 RCM 负责）
        """
        B, Cq, Hq, Wq = Q.shape
        _, Ck, Hk, Wk = K.shape
        _, Cv, Hv, Wv = V.shape

        if (Hq != Hk) or (Wq != Wk):
            raise RuntimeError(f"Q 与 K 的空间尺寸必须一致: Q=({Hq},{Wq}), K=({Hk},{Wk})")

        # Flatten Q, K -> (B, Nq, C)
        Nq = Hq * Wq
        q_flat = Q.permute(0, 2, 3, 1).reshape(B, Nq, Cq)  # (B, Nq, C)
        k_flat = K.permute(0, 2, 3, 1).reshape(B, Nq, Ck)  # (B, Nq, C)

        # 若通道数不一致，则尝试对 K 或 Q 做简单线性对齐（通常不会发生）
        if Ck != Cq:
            # 将 k_flat 投影到 Cq（使用一个临时线性层）
            k_flat = k_flat.reshape(B * Nq, Ck)
            proj_k = nn.Linear(Ck, Cq).to(k_flat.device)
            k_flat = proj_k(k_flat).view(B, Nq, Cq)

        # 相似度矩阵： (B, Nq, Nq) = q_flat @ k_flat^T  并除以 di
        attn_logits = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / (self.di + self.eps)
        attn = F.softmax(attn_logits, dim=-1)  # (B, Nq, Nq)

        # 处理 V：若空间尺寸与 Q 不同，则采样到 Q 的空间尺寸
        if (Hv != Hq) or (Wv != Wq):
            V_resized = F.interpolate(V, size=(Hq, Wq), mode=self.interp_mode, align_corners=False if self.interp_mode == 'bilinear' else None)
        else:
            V_resized = V

        # 若 V 通道数与 Q 不一致，则用内部 1x1 投影或动态创建投影
        if V_resized.shape[1] != Cq:
            if (self._v_proj is None) or (self._v_proj.out_channels != Cq) or (self._v_proj.in_channels != V_resized.shape[1]):
                self._v_proj = nn.Conv2d(V_resized.shape[1], Cq, kernel_size=1, bias=False).to(V_resized.device)
            V_resized = self._v_proj(V_resized)

        # 展平 V -> (B, Nq, Cq)
        v_flat = V_resized.permute(0, 2, 3, 1).reshape(B, Nq, Cq)

        # attention 权重乘以 V -> (B, Nq, Cq)
        out_flat = torch.matmul(attn, v_flat)

        # 恢复为 (B, Cq, Hq, Wq)
        out = out_flat.view(B, Hq, Wq, Cq).permute(0, 3, 1, 2)

        # LayerNorm 按通道：先转为 (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.layernorm(out)
        out = out.permute(0, 3, 1, 2)

        return out


class SES(nn.Module):
    """超分辨率增强方案 (SES) - 论文3.4节"""

    def __init__(self, channels: int, scale_factor: int = 2):
        super(SES, self).__init__()
        self.up = nn.Sequential(
            # 使用转置卷积直接上采样 s(scale factor) 倍
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                padding=0,
                output_padding=0,
                bias=False
            ),
            nn.ReLU(),
            # 后续卷积用于特征精炼
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)  # 输出: (B, C, s*H, s*W)

class FFN(nn.Module):
    """下采样版前馈网络 (FFN) - 用于下采样 RCM（尺度减半）
       1x1 -> GELU -> depthwise 3x3 (stride=2) -> GELU -> 1x1
    """
    def __init__(self, channels: int, mult: int = 4):
        super(FFN, self).__init__()
        hidden = channels * mult
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=False, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入/输出均为 [B, C, H, W]
        return self.net(x)


class FFN_Same(nn.Module):
    """保持尺寸不变的前馈网络（用于第5个RCM）
       1x1 -> GELU -> depthwise 3x3 (stride=1) -> GELU -> 1x1
    """
    def __init__(self, channels: int, mult: int = 4):
        super(FFN_Same, self).__init__()
        hidden = channels * mult
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFN_Upsample(nn.Module):
    """上采样版前馈网络 (FFN) - 用于上采样 RCM（尺度加倍）
       先上采样，再执行 1x1 -> GELU -> depthwise 3x3 (stride=1) -> GELU -> 1x1
    """
    def __init__(self, channels: int, mult: int = 4, scale_factor: int = 2):
        super(FFN_Upsample, self).__init__()
        hidden = channels * mult
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.net(x)

class RCM(nn.Module):
    """残差缓解与组件增强模块 (RCM) - 支持 down / same / up 三种模式"""
    def __init__(self, channels: int, mode: str = 'down'):
        super(RCM, self).__init__()
        self.ses = SES(channels)
        self.mres_l = MRES(channels)
        self.mres_r = MRES(channels)
        # 根据 mode 选择不同的 FFN 实现
        if mode == 'up':
            self.ffn_l = FFN_Upsample(channels)
            self.ffn_r = FFN_Upsample(channels)
        elif mode == 'same':
            self.ffn_l = FFN_Same(channels)
            self.ffn_r = FFN_Same(channels)
        else:  # 'down' or default
            self.ffn_l = FFN(channels)
            self.ffn_r = FFN(channels)

        # 等价于对每个空间位置按通道做 LayerNorm
        # self.norm_l = nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-5, affine=True)
        # self.norm_r = nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-5, affine=True)
        self.norm_l = nn.LayerNorm(64)
        self.norm_r = nn.LayerNorm(64)

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
        # L_enhanced = self.norm_l(L_enhanced)
        # 将 (N,C,H,W) -> (N,H,W,C)，对最后一维通道做 LayerNorm，再换回来
        L_enhanced = L_enhanced.permute(0, 2, 3, 1)
        L_enhanced = self.norm_l(L_enhanced)
        L_enhanced = L_enhanced.permute(0, 3, 1, 2)

        R_enhanced = self.ffn_r(residual_R + R)
        # R_enhanced = self.norm_r(R_enhanced)
        R_enhanced = R_enhanced.permute(0, 2, 3, 1)
        R_enhanced = self.norm_r(R_enhanced)
        R_enhanced = R_enhanced.permute(0, 3, 1, 2)

        return L_enhanced, R_enhanced

class FeatureExtractor(nn.Module):
    """特征提取器：将3通道图像转换为高维特征"""
    def __init__(self, in_channels=3, out_channels=64):
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
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

# class IRetinex(nn.Module):
#     """9个RCM构成 U 型网络，前4下采样 + 第5保持尺寸 + 后4上采样；
#        后4个 RCM 输入采用跳跃相加"""
#
#     def __init__(self, feature_channels=64):
#         super(IRetinex, self).__init__()
#         self.dual_color = DualColorSpacePrior()
#         self.reflectance_decomp = ReflectanceDecomposition()
#
#         # 特征提取器：3通道→64通道
#         self.feature_extractor_l = FeatureExtractor(3, feature_channels)
#         self.feature_extractor_r = FeatureExtractor(3, feature_channels)
#
#         # RCM模块堆叠：前4个下采样、1个保持尺寸、后4个上采样
#         self.rcm_down = nn.ModuleList([RCM(feature_channels, mode='down') for _ in range(4)])  # RCM1-4
#         self.rcm_mid = RCM(feature_channels, mode='same')  # RCM5（保持尺寸）
#         self.rcm_up = nn.ModuleList([RCM(feature_channels, mode='up') for _ in range(4)])  # RCM6-9
#
#         # 特征重建器：64通道→3通道
#         self.feature_reconstructor_l = FeatureReconstructor(feature_channels, 3)
#         self.feature_reconstructor_r = FeatureReconstructor(feature_channels, 3)
#
#     def forward(self, x: torch.Tensor) -> tuple:
#         # 1. ICRR模块初始化L和R（3通道）
#         L_init = self.dual_color(x)
#         R_init = self.reflectance_decomp(x, L_init)
#
#         # 2. 特征提取：3通道→64通道高维特征
#         L_feat = self.feature_extractor_l(L_init)
#         R_feat = self.feature_extractor_r(R_init)
#
#         # 3. 执行 RCM：先 RCM1-4（下采样），再 RCM5（保持），再 RCM6-9（上采样，带跳跃相加）
#         all_L_feats = []  # 存储 RCM1..9 的 L 输出
#         all_R_feats = []
#
#         # RCM1-4（down）
#         for rcm in self.rcm_down:
#             L_feat, R_feat = rcm(L_feat, R_feat)
#             all_L_feats.append(L_feat)
#             all_R_feats.append(R_feat)
#
#         # RCM5（same）
#         L_feat, R_feat = self.rcm_mid(L_feat, R_feat)
#         all_L_feats.append(L_feat)  # index 4 => RCM5 输出
#         all_R_feats.append(R_feat)
#
#         # RCM6-9（up） — 使用跳跃相加输入
#         # RCM6 输入 = RCM4 + RCM5
#         L_in = all_L_feats[3] + all_L_feats[4]
#         R_in = all_R_feats[3] + all_R_feats[4]
#         L_feat, R_feat = self.rcm_up[0](L_in, R_in)
#         all_L_feats.append(L_feat)  # index 5 => RCM6
#         all_R_feats.append(R_feat)
#
#         # RCM7 输入 = RCM3 + RCM6
#         L_in = all_L_feats[2] + all_L_feats[5]
#         R_in = all_R_feats[2] + all_R_feats[5]
#         L_feat, R_feat = self.rcm_up[1](L_in, R_in)
#         all_L_feats.append(L_feat)  # index 6 => RCM7
#         all_R_feats.append(R_feat)
#
#         # RCM8 输入 = RCM2 + RCM7
#         L_in = all_L_feats[1] + all_L_feats[6]
#         R_in = all_R_feats[1] + all_R_feats[6]
#         L_feat, R_feat = self.rcm_up[2](L_in, R_in)
#         all_L_feats.append(L_feat)  # index 7 => RCM8
#         all_R_feats.append(R_feat)
#
#         # RCM9 输入 = RCM1 + RCM8
#         L_in = all_L_feats[0] + all_L_feats[7]
#         R_in = all_R_feats[0] + all_R_feats[7]
#         L_feat, R_feat = self.rcm_up[3](L_in, R_in)
#         all_L_feats.append(L_feat)  # index 8 => RCM9 (最终尺度：与输入相同)
#         all_R_feats.append(R_feat)
#
#         # 4. 提取后5个RCM输出（RCM5..RCM9）
#         L_list = all_L_feats[4:]  # 索引4-8（共5个）
#         R_list = all_R_feats[4:]
#
#         # 5. 使用最后一个 RCM 的输出直接重建（不再有额外 final_upsample）
#         L_final = self.feature_reconstructor_l(all_L_feats[-1])
#         R_final = self.feature_reconstructor_r(all_R_feats[-1])
#
#         # 6. 最终增强图像（与输入相同分辨率）
#         enhanced = L_final * R_final
#
#         # 返回：保持接口不变
#         return enhanced, L_init, R_init, L_final, R_final, L_list, R_list


class IRetinex(nn.Module):
    """4 层次 U-Net 版本的 IRetinex：3 下采样 + 1 保持 + 3 上采样（共 7 个 RCM）"""
    def __init__(self, feature_channels=64):
        super(IRetinex, self).__init__()
        self.dual_color = DualColorSpacePrior()
        self.reflectance_decomp = ReflectanceDecomposition()

        # 特征提取器：3通道→feature_channels
        self.feature_extractor_l = FeatureExtractor(3, feature_channels)
        self.feature_extractor_r = FeatureExtractor(3, feature_channels)

        # RCM 模块堆叠：前3个下采样、1个保持、后3个上采样 -> 共7个 RCM
        self.rcm_down = nn.ModuleList([RCM(feature_channels, mode='down') for _ in range(3)])  # RCM1-3
        self.rcm_mid = RCM(feature_channels, mode='same')  # RCM4（保持尺寸）
        self.rcm_up = nn.ModuleList([RCM(feature_channels, mode='up') for _ in range(3)])  # RCM5-7

        # 特征重建器：feature_channels -> 3
        self.feature_reconstructor_l = FeatureReconstructor(feature_channels, 3)
        self.feature_reconstructor_r = FeatureReconstructor(feature_channels, 3)

    def forward(self, x: torch.Tensor) -> tuple:
        # 1. ICRR 模块初始化 L 和 R（3 通道）
        L_init = self.dual_color(x)
        R_init = self.reflectance_decomp(x, L_init)

        # 2. 特征提取：3 通道 -> feature_channels
        L_feat = self.feature_extractor_l(L_init)
        R_feat = self.feature_extractor_r(R_init)

        # 3. 执行 RCM：先 RCM1-3（下采样），再 RCM4（保持），再 RCM5-7（上采样，带跳跃相加）
        all_L_feats = []  # 存储 RCM1..7 的 L 输出
        all_R_feats = []

        # RCM1-3（down）
        for rcm in self.rcm_down:
            L_feat, R_feat = rcm(L_feat, R_feat)
            all_L_feats.append(L_feat)
            all_R_feats.append(R_feat)

        # RCM4（same）
        L_feat, R_feat = self.rcm_mid(L_feat, R_feat)
        all_L_feats.append(L_feat)  # index 3 => RCM4 输出
        all_R_feats.append(R_feat)

        # 上采样阶段（RCM5-7） — 使用对称跳跃相加
        # RCM5 输入 = RCM3 + RCM4
        L_in = all_L_feats[2] + all_L_feats[3]
        R_in = all_R_feats[2] + all_R_feats[3]
        L_feat, R_feat = self.rcm_up[0](L_in, R_in)
        all_L_feats.append(L_feat)  # index 4 => RCM5
        all_R_feats.append(R_feat)

        # RCM6 输入 = RCM2 + RCM5
        L_in = all_L_feats[1] + all_L_feats[4]
        R_in = all_R_feats[1] + all_R_feats[4]
        L_feat, R_feat = self.rcm_up[1](L_in, R_in)
        all_L_feats.append(L_feat)  # index 5 => RCM6
        all_R_feats.append(R_feat)

        # RCM7 输入 = RCM1 + RCM6
        L_in = all_L_feats[0] + all_L_feats[5]
        R_in = all_R_feats[0] + all_R_feats[5]
        L_feat, R_feat = self.rcm_up[2](L_in, R_in)
        all_L_feats.append(L_feat)  # index 6 => RCM7 (最终尺度：与输入相同)
        all_R_feats.append(R_feat)

        # 4. 提取后 4 个 RCM 输出（RCM4..RCM7）
        L_list = all_L_feats[3:]  # 索引3-6（共4个）
        R_list = all_R_feats[3:]

        # 5. 使用最后一个 RCM 的输出直接重建（不再有额外 final_upsample）
        L_final = self.feature_reconstructor_l(all_L_feats[-1])
        R_final = self.feature_reconstructor_r(all_R_feats[-1])

        # 6. 最终增强图像（与输入相同分辨率）
        enhanced = L_final * R_final

        # 返回：保持接口不变
        return enhanced, L_init, R_init, L_final, R_final, L_list, R_list
