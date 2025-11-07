import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
#from networks.SKModule_3D import DeformBasicBlock



import torch
import torch.nn as nn
from torchvision import datasets, transforms
    




# ===== FEATURE RECTIFIER (3D VERSION) =====
# Input/Output shape: (B, C, D, H, W)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SKLayer3D(nn.Module):
    def __init__(self, planes: int, ratio: int = 16):
        super().__init__()
        d = max(planes // ratio, 32)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.LayerNorm(d),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)

    def forward(self, u1, u2, x):
        B, C, D, H, W = u1.shape
        u = u1 + u2 + x
        s = self.avgpool(u).flatten(1)                # (B, C)
        z = self.fc(s)                                # (B, d)
        a_b = torch.stack([self.fc1(z), self.fc2(z)], dim=1)  # (B, 2, C)
        w = a_b.softmax(dim=1)
        a = w[:, 0].view(B, C, 1, 1, 1)
        b = w[:, 1].view(B, C, 1, 1, 1)
        return u1 * a + u2 * b

class DepthwiseSeparable3D(nn.Module):
    def __init__(self, C: int, k: int = 3, r: int = 2):
        super().__init__()
        assert k % 2 == 1
        pad = k // 2
        Cmid = max(C // r, 1)
        self.reduce = nn.Conv3d(C, Cmid, 1, bias=True)
        self.dw     = nn.Conv3d(Cmid, Cmid, k, padding=pad, groups=Cmid, bias=True)
        self.expand = nn.Conv3d(Cmid, C, 1, bias=True)
        self.act    = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.expand(self.act(self.dw(self.reduce(x))))

class UGFR(nn.Module):
    def __init__(self, in_channels: int, k: int = 3, bottleneck_ratio: int = 2, use_trilinear: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=True)

        # pyramids
        self.p2 = nn.AvgPool3d(2)     # /2
        self.p4 = nn.AvgPool3d(2)     # apply on p2() to get /4 (i.e., p4(p2(x)))

        up_mode = 'trilinear' if use_trilinear else 'nearest'
        self.up2 = nn.Upsample(scale_factor=(2,2,2), mode=up_mode, align_corners=False if use_trilinear else None)
        self.up4 = nn.Upsample(scale_factor=(4,4,4), mode=up_mode, align_corners=False if use_trilinear else None)

        # depthwise-separable per scale/branch
        self.dw_ctx_s1 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)
        self.dw_ctx_s2 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)
        self.dw_ctx_s4 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)

        self.dw_det_s1 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)
        self.dw_det_s2 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)
        self.dw_det_s4 = DepthwiseSeparable3D(in_channels, k=k, r=bottleneck_ratio)

        self.sk   = SKLayer3D(in_channels)
        self.post = nn.Sequential(
            nn.GroupNorm(8 if in_channels % 8 == 0 else 1, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, uncertainty):
        # seeds
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # context pyramid (down then up)
        d2 = self.p2(x1)          # /2
        d4 = self.p4(d2)          # /4 (i.e., /2 of /2)

        ctx_s1 = self.dw_ctx_s1(x1 + uncertainty)     # base
        ctx_s2 = self.up2(self.dw_ctx_s2(d2))         # /2 -> base
        ctx_s4 = self.up4(self.dw_ctx_s4(d4))         # /4 -> base
        x1_context = ctx_s1 + ctx_s2 + ctx_s4

        # detail pyramid (up then down)
        u2 = self.up2(x2)         # ×2
        u4 = self.up4(x2)         # ×4

        det_s1 = self.dw_det_s1(x2 + uncertainty)                # base
        det_s2 = self.p2(self.dw_det_s2(u2))                      # ×2 -> base
        det_s4 = self.p2(self.p2(self.dw_det_s4(u4)))             # ×4 -> /2 -> base  
        x2_detail = det_s1 + det_s2 + det_s4

        y = self.sk(x1_context, x2_detail, x)
        return self.post(y)


"""
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32, 32)
    attn = SKLayer(3, 3)
    y = attn(x,uncertainty=None)
    print(y.shape)
"""

