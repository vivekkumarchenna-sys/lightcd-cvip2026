"""Losses for binary change detection under heavy class imbalance (~4.5% positive).

Combined weighted-BCE + soft-Dice on the change class. This is the standard,
robust choice for sparse change masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.reshape(prob.size(0), -1)
        target = target.reshape(target.size(0), -1)
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """weighted BCE + Dice. pos_weight up-weights the sparse change class."""

    def __init__(self, pos_weight=None, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.dice = DiceLoss()
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, target):
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        dl = self.dice(logits, target)
        return self.bce_w * bce + self.dice_w * dl
