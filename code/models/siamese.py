"""Generic Siamese-encoder U-Net for change detection with a swappable
bi-temporal fusion module. Backbone comes from timm (features_only), so we can
instantiate an efficient LightCD (MobileNetV3) or a heavier ResNet18 baseline
from the same code.

Fusion modes (per scale):
  - 'concat' : 1x1( [f_A ; f_B] )                       (simple baseline fusion)
  - 'diff'   : 1x1( |f_A - f_B| )                       (difference baseline)
  - 'tda'    : Temporal-Difference Attention (ours)     (difference-guided
               channel + spatial attention over the fused feature)

TDA is the paper's contribution: it fuses [f_A ; f_B ; |f_A-f_B|] then gates the
result with channel attention (SE computed FROM the temporal difference) and a
difference-driven spatial attention map. Both gates are cheap, keeping the model
edge-friendly while focusing capacity on genuinely changed regions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SEFromDiff(nn.Module):
    """Channel attention whose weights are computed from the temporal difference."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(4, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, diff):
        return self.fc(diff)


class SpatialAttnFromDiff(nn.Module):
    """Spatial attention map from the temporal difference (mean+max pooled)."""

    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, diff):
        avg = diff.mean(dim=1, keepdim=True)
        mx, _ = diff.max(dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(a))


class TDAFusion(nn.Module):
    def __init__(self, ch, mode="tda", use_ca=True, use_sa=True, reduction=8):
        super().__init__()
        assert mode in ("tda", "diff", "concat")
        self.mode = mode
        self.use_ca = use_ca
        self.use_sa = use_sa
        if mode == "concat":
            fuse_in = 2 * ch
        elif mode == "diff":
            fuse_in = ch
        else:  # tda
            fuse_in = 3 * ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        if mode == "tda":
            if use_ca:
                self.ca = SEFromDiff(ch, reduction)
            if use_sa:
                self.sa = SpatialAttnFromDiff()

    def forward(self, fa, fb):
        diff = torch.abs(fa - fb)
        if self.mode == "concat":
            x = self.fuse(torch.cat([fa, fb], 1))
        elif self.mode == "diff":
            x = self.fuse(diff)
        else:
            x = self.fuse(torch.cat([fa, fb, diff], 1))
            if self.use_ca:
                x = x * self.ca(diff)
            if self.use_sa:
                x = x * self.sa(diff)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseUNet(nn.Module):
    def __init__(self, backbone="mobilenetv3_small_100", fusion="tda",
                 use_ca=True, use_sa=True, pretrained=True, n_class=1, output_stride=32):
        super().__init__()
        kw = {}
        if output_stride != 32:
            kw["output_stride"] = output_stride  # less aggressive downsampling (dilation)
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, **kw
        )
        chs = self.encoder.feature_info.channels()  # e.g. [16,16,24,48,576]
        self.chs = chs
        self.fusions = nn.ModuleList(
            [TDAFusion(c, mode=fusion, use_ca=use_ca, use_sa=use_sa) for c in chs]
        )
        # U-Net decoder from deepest to shallowest
        self.dec_blocks = nn.ModuleList()
        prev = chs[-1]
        for i in range(len(chs) - 2, -1, -1):
            self.dec_blocks.append(DecoderBlock(prev + chs[i], chs[i]))
            prev = chs[i]
        self.head = nn.Conv2d(chs[0], n_class, 1)

    def forward(self, a, b):
        fa = self.encoder(a)
        fb = self.encoder(b)
        fused = [fuse(x, y) for fuse, x, y in zip(self.fusions, fa, fb)]
        x = fused[-1]
        j = 0
        for i in range(len(fused) - 2, -1, -1):
            x = F.interpolate(x, size=fused[i].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, fused[i]], 1)
            x = self.dec_blocks[j](x)
            j += 1
        x = F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)
