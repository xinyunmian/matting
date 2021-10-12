
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.loss.multi_task import MultiTaskProxy
from core.loss.segment import SemanticSegmentation
from core.loss.matting import ImageMatting
from core.loss.grad import MattingGrad, ImageGradient


class JointMattingParsingLoss(nn.Module):
    def __init__(self, phase):
        super(JointMattingParsingLoss, self).__init__()
        assert phase in ['pretrain', 'end2end']
        print('============> Using join loss: {}'.format(phase))
        self.phase = phase
        self._multi_task_weight(phase)
        self.segment_loss = SemanticSegmentation()
        self.matting_loss = ImageMatting()
        # self.gradient_loss = MattingGrad()

    def _multi_task_weight(self, phase):
        if phase == 'end2end':
            self.proxy = MultiTaskProxy(num_task=2)

    def forward(self, target, output):
        segment_pr, matting_pr = output['segment'], output['matting']
        segment_gt, matting_gt = target['segment'], target['matting']

        loss_segment = self.segment_loss(segment_pr=segment_pr, segment_gt=segment_gt, weight=None)

        if self.phase == 'pretrain':
            loss = loss_segment
            return loss
        else:
            loss_matting = self.matting_loss(matting_pr=matting_pr, matting_gt=matting_gt, segment_gt=segment_gt)
            loss = self.proxy([loss_segment, loss_matting])
            return loss, loss_segment, loss_matting


