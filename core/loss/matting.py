
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageMatting(nn.Module):
    def __init__(self):
        super(ImageMatting, self).__init__()

    def _forward_in_binary(self, matting_pr, matting_gt, segment_gt):
        # just for binary classification
        with torch.no_grad():
            num_pos = torch.mean(segment_gt.float(), dim=(1, 2, 3)).view(-1, 1, 1, 1)
            num_neg = 1. - num_pos
            beta = torch.ones_like(segment_gt, dtype=torch.float32) * num_neg
            alpha = torch.ones_like(segment_gt, dtype=torch.float32) * num_pos
            weight = torch.where(segment_gt == 1, beta, alpha).float()

        loss = F.l1_loss(matting_pr, matting_gt, reduction='none') * weight
        return loss.mean()

    def _forward_in_tri(self, matting_pr, matting_gt, segment_gt):
        with torch.no_grad():
            transition = torch.where(segment_gt == 1, torch.ones_like(segment_gt), torch.zeros_like(segment_gt)).float()
            num_pos = torch.sum(transition, dim=(1, 2))
            num_neg = torch.sum(1 - transition, dim=(1, 2))
            beta = (num_neg / (num_pos + num_neg)).view(-1, 1, 1)
            alpha = (1. - beta).view(-1, 1, 1)
            weight = torch.where(segment_gt == 1, torch.ones_like(segment_gt).float() * beta, torch.ones_like(segment_gt).float() * alpha).float()

        l1 = F.l1_loss(matting_pr, matting_gt, reduction='none')
        l2 = F.mse_loss(matting_pr, matting_gt, reduction='none')
        loss = torch.where(transition == 1, l2, l1) * weight
        return loss.mean()

    def _forward_in(self, matting_pr, matting_gt, segment_gt):
        N, _, H, W = list(matting_pr.shape)
        matting_gt = F.interpolate(matting_gt, size=(H, W), mode='bilinear', align_corners=True)
        matting_pr = matting_pr.view(N, 1, H, W)
        matting_gt = matting_gt.view(N, 1, H, W)
        loss = F.l1_loss(matting_pr, matting_gt, reduction='mean')
        return loss

    def forward(self, matting_pr, matting_gt, segment_gt):
        if isinstance(matting_pr, list):
            loss = list()
            for pr in matting_pr:
                loss.append(self._forward_in_binary(matting_pr=pr, matting_gt=matting_gt, segment_gt=segment_gt))
            return sum(loss)
        else:
            return self._forward_in(matting_pr=matting_pr, matting_gt=matting_gt, segment_gt=segment_gt)