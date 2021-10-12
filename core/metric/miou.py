
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanInterSectionOverUnion(nn.Module):
    def __init__(self, num_classes):
        super(MeanInterSectionOverUnion, self).__init__()
        self.num_classes = num_classes

    def _confusion_matrix(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)

        target_mask = (target >= 0) & (target < self.num_classes)
        target = target[target_mask]
        predict = predict[target_mask]

        indices = self.num_classes * target + predict
        return torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def _extract_segment(self, target, output):
        output = torch.argmax(output, dim=1)
        return output, target

    def _forward_in(self, target, output):
        segment_pr, segment_gt = self._extract_segment(target, output)
        confusion_matrix = self._confusion_matrix(segment_pr, segment_gt)
        sum_row = torch.sum(confusion_matrix, dim=0).float()
        sum_col = torch.sum(confusion_matrix, dim=1).float()
        diagonal = torch.diagonal(confusion_matrix, 0).float()
        denominator = (sum_row + sum_col - diagonal)
        num_valid_entries = torch.sum(denominator > 0)
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        iou = diagonal / denominator
        return torch.sum(iou) / num_valid_entries

    def forward(self, target, output):
        segment_pr = output['segment']
        segment_gt = target['segment']
        with torch.no_grad():
            if isinstance(segment_pr, list):
                miou = list()
                for pr in segment_pr:
                    N, C, H, W = list(pr.shape)
                    gt = F.interpolate(segment_gt.float(), size=(H, W), mode='nearest').long()
                    miou.append(self._forward_in(target=gt, output=pr))
                return sum(miou) / len(segment_pr)
            else:
                return self._forward_in(target=segment_gt, output=segment_pr)
