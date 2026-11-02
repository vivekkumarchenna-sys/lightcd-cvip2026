"""TinyCD baseline (Codegoni et al., Neural Comput. Appl. 2023), ported from the
authors' repository github.com/AndreaCodegoni/Tiny_model_4_CD.

The structure is unchanged: the first four EfficientNet-B4 stages, mixing and
mask-attention blocks, and the up-sampling decoder. The only edit is the output:
the final Sigmoid is dropped so the model returns logits for our shared loss.

Research and non-commercial use only, as the source repository states. Not covered by the
MIT licence of this repository.
"""
from typing import List

import torch
import torch.nn as nn
import torchvision


class PixelwiseLinear(nn.Module):
    def __init__(self, fin: List[int], fout: List[int], last_activation=None):
        super().__init__()
        n = len(fin)
        self._linears = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                nn.PReLU() if i < n - 1 or last_activation is None else last_activation,
            )
            for i in range(n)
        ])

    def forward(self, x):
        return self._linears(x)


class MixingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x, y):
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self._convmix(mixed)


class MixingMaskAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out, fin, fout):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)

    def forward(self, x, y):
        return self._linear(self._mixing(x, y))


class UpMask(nn.Module):
    def __init__(self, scale_factor, nin, nout):
        super().__init__()
        self._upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self._convolution = nn.Sequential(
            nn.Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nin),
            nn.Conv2d(nin, nout, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nout),
        )

    def forward(self, x, y=None):
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class TinyCD(nn.Module):
    def __init__(self, pretrained=True, output_layer_bkbn="3"):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        entire = torchvision.models.efficientnet_b4(weights=weights).features
        self._backbone = nn.ModuleList([])
        for name, layer in entire.named_children():
            self._backbone.append(layer)
            if name == output_layer_bkbn:
                break
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = nn.ModuleList([
            MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
            MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
            MixingBlock(112, 56),
        ])
        self._up = nn.ModuleList([UpMask(2, 56, 64), UpMask(2, 64, 64), UpMask(2, 64, 32)])
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], nn.Identity())

    def forward(self, a, b):
        features = [self._first_mix(a, b)]
        for num, layer in enumerate(self._backbone):
            a, b = layer(a), layer(b)
            if num != 0:
                features.append(self._mixing_mask[num - 1](a, b))
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return self._classify(upping)
