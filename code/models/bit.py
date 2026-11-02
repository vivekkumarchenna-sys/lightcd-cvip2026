"""BIT baseline (Chen et al., IEEE TGRS 2022), ported from the authors' repository
github.com/justchenhao/BIT_CD, configuration base_transformer_pos_s4_dd8_dedim8
(the BIT_LEVIR model: ResNet-18 to stage 4, four tokens, one encoder layer, eight
decoder layers). Output is a single logit map for our shared loss instead of two
softmax channels; ImageNet weights come from the timm ResNet-18 used elsewhere.

Research and non-commercial use only, as the source repository states. Not covered by the
MIT licence of this repository.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(x, **kw) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kw):
        return self.fn(x, x2, **kw) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kw):
        return self.fn(self.norm(x), self.norm(x2), **kw)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn = (torch.einsum("bhid,bhjd->bhij", q, k) * self.scale).softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim))

    def forward(self, x, m):
        q, k, v = self.to_q(x), self.to_k(m), self.to_v(m)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])
        attn = (torch.einsum("bhid,bhjd->bhij", q, k) * self.scale).softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([
            Residual(PreNorm(dim, Attention(dim, heads, dim_head))),
            Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
        ]) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = ff(attn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([
            Residual2(PreNorm2(dim, CrossAttention(dim, heads, dim_head))),
            Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
        ]) for _ in range(depth)])

    def forward(self, x, m):
        for attn, ff in self.layers:
            x = ff(attn(x, m))
        return x


class BIT(nn.Module):
    def __init__(self, pretrained=True, token_len=4, dim=32, enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=8):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        if pretrained:
            sd = timm.create_model("resnet18", pretrained=True).state_dict()
            self.resnet.load_state_dict(sd, strict=False)
        # stride 8 with dilation in stage 3, as replace_stride_with_dilation does in
        # the authors' ResNet (current torchvision refuses it for BasicBlock)
        for i, block in enumerate(self.resnet.layer3):
            d = 1 if i == 0 else 2
            block.conv1.stride, block.conv1.dilation, block.conv1.padding = (1, 1), (d, d), (d, d)
            block.conv2.dilation, block.conv2.padding = (2, 2), (2, 2)
        self.resnet.layer3[0].downsample[0].stride = (1, 1)
        del self.resnet.layer4, self.resnet.fc      # stage 4 is the last one used
        self.conv_pred = nn.Conv2d(256, dim, 3, padding=1)
        self.conv_a = nn.Conv2d(dim, token_len, 1, bias=False)
        self.token_len = token_len
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, dim))
        self.transformer = Transformer(dim, enc_depth, 8, dim_head, 2 * dim)
        self.transformer_decoder = TransformerDecoder(dim, dec_depth, 8, decoder_dim_head, 2 * dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(dim, 1, 3, padding=1))

    def _single(self, x):
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x = r.layer3(r.layer2(r.layer1(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")   # stride 8 -> 4
        return self.conv_pred(x)

    def _tokens(self, x):
        b, c, _, _ = x.shape
        a = self.conv_a(x).view(b, self.token_len, -1).softmax(dim=-1)
        return torch.einsum("bln,bcn->blc", a, x.view(b, c, -1))

    def _decode(self, x, m):
        _, _, h, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer_decoder(x, m)
        return rearrange(x, "b (h w) c -> b c h w", h=h)

    def forward(self, a, b):
        x1, x2 = self._single(a), self._single(b)
        tokens = torch.cat([self._tokens(x1), self._tokens(x2)], dim=1) + self.pos_embedding
        t1, t2 = self.transformer(tokens).chunk(2, dim=1)
        x = torch.abs(self._decode(x1, t1) - self._decode(x2, t2))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.classifier(x)
