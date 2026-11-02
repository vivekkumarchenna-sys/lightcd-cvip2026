"""Classic fully-convolutional change-detection baselines (Daudt et al., 2018):
FC-EF (early fusion), FC-Siam-conc, FC-Siam-diff. Lightweight, from scratch.

Reimplemented compactly with a shared U-Net skeleton and a configurable
fusion mode so all three share one tested code path.
"""
import torch
import torch.nn as nn


def cbr(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(cbr(cin, cout), cbr(cout, cout))

    def forward(self, x):
        return self.net(x)


class FCSiam(nn.Module):
    """Unified FC-EF / FC-Siam-conc / FC-Siam-diff.

    mode: 'ef' (early fusion, single encoder over concatenated 6ch input),
          'conc' (siamese encoder, skip = concat of A,B features),
          'diff' (siamese encoder, skip = |A-B| features).
    """

    def __init__(self, mode="diff", chs=(16, 32, 64, 128), in_ch=3, n_class=1):
        super().__init__()
        assert mode in ("ef", "conc", "diff")
        self.mode = mode
        self.pool = nn.MaxPool2d(2)
        enc_in = in_ch * 2 if mode == "ef" else in_ch

        # encoder
        self.enc = nn.ModuleList()
        prev = enc_in
        for c in chs:
            self.enc.append(DoubleConv(prev, c))
            prev = c
        self.bott = DoubleConv(chs[-1], chs[-1] * 2)

        # decoder
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        skip_mult = 2 if mode == "conc" else 1
        dprev = chs[-1] * 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(dprev, c, 2, stride=2))
            self.dec.append(DoubleConv(c + c * skip_mult, c))
            dprev = c
        self.head = nn.Conv2d(chs[0], n_class, 1)

    def _encode(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        return x, skips

    def forward(self, a, b):
        if self.mode == "ef":
            x, skips = self._encode(torch.cat([a, b], 1))
            fused = skips
        else:
            xa, sa = self._encode(a)
            xb, sb = self._encode(b)
            x = torch.abs(xa - xb) if self.mode == "diff" else (xa + xb) * 0.5
            if self.mode == "diff":
                fused = [torch.abs(fa - fb) for fa, fb in zip(sa, sb)]
            else:
                fused = [torch.cat([fa, fb], 1) for fa, fb in zip(sa, sb)]
        for up, dec, skip in zip(self.ups, self.dec, reversed(fused)):
            x = up(x)
            # pad if needed
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)
