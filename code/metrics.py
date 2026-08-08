"""Streaming binary-change-detection metrics.

Accumulates TP/FP/FN/TN over batches, then reports the change-class
Precision, Recall, F1, IoU, plus Overall Accuracy and Cohen's Kappa.
These are the standard CD benchmark metrics (LEVIR-CD, WHU-CD, etc.).
"""
import torch


class CDMetrics:
    def __init__(self, threshold=0.5):
        self.t = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits, target):
        pred = (torch.sigmoid(logits) > self.t).long()
        target = target.long()
        self.tp += int(((pred == 1) & (target == 1)).sum())
        self.fp += int(((pred == 1) & (target == 0)).sum())
        self.fn += int(((pred == 0) & (target == 1)).sum())
        self.tn += int(((pred == 0) & (target == 0)).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        total = tp + fp + fn + tn
        oa = (tp + tn) / (total + eps)
        # Cohen's kappa (binary)
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total + eps)
        kappa = (oa - pe) / (1 - pe + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "oa": oa,
            "kappa": kappa,
        }
