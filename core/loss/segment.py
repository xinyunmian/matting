
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentation(nn.Module):
    def __init__(self):
        super(SemanticSegmentation, self).__init__()

    def _forward_in(self, segment_pr, segment_gt, weight=None):
        N, _, H, W = list(segment_pr.shape)
        segment_gt = F.interpolate(segment_gt.float(), size=(H, W), mode='nearest').long()
        segment_pr = segment_pr.view(N, -1, H, W)
        segment_gt = segment_gt.view(N, H, W)
        if weight is not None:
            weight = weight.view(N, H, W)
            batch_loss = F.cross_entropy(segment_pr, segment_gt, reduction='none') * weight
            loss = batch_loss.mean()
        else:
            loss = F.cross_entropy(segment_pr, segment_gt, reduction='mean')
        return loss

    def forward(self, segment_pr, segment_gt, weight=None):
        if isinstance(segment_pr, list):
            loss = list()
            for pr in segment_pr:
                loss.append(self._forward_in(segment_pr=pr, segment_gt=segment_gt, weight=weight))
            return sum(loss)
        else:
            return self._forward_in(segment_pr=segment_pr, segment_gt=segment_gt, weight=weight)