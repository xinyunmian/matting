
import torch
import torch.nn as nn
import torch.nn.functional as F


class MattingGrad(nn.Module):
    def __init__(self):
        super(MattingGrad, self).__init__()
        weight_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()
        weight_y = torch.tensor([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]]).float()
        self.weight_x = nn.Parameter(weight_x.view((1, 1, 3, 3)), requires_grad=False)
        self.weight_y = nn.Parameter(weight_y.view((1, 1, 3, 3)), requires_grad=False)

    def calculate_gradient(self, image):
        # image value show be in [0,1]
        gray = torch.mean(image * 255, dim=1, keepdim=True)
        x = torch.abs(F.conv2d(gray, weight=self.weight_x, stride=1, padding=1, dilation=1))
        y = torch.abs(F.conv2d(gray, weight=self.weight_y, stride=1, padding=1, dilation=1))
        x = torch.clamp(x, 0, 255)
        y = torch.clamp(y, 0, 255)
        return torch.round(0.5 * x + 0.5 * y) / 255.

    def forward(self, matting_pr, matting_gt):
        grad_pr = self.calculate_gradient(matting_pr)
        grad_gt = self.calculate_gradient(matting_gt)
        return F.l1_loss(input=grad_pr, target=grad_gt, reduction='mean')


class ImageGradient(nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()

    def gradient(self, image):
        gx = image[:, :, :, 1:] - image[:, :, :, :-1]
        gy = image[:, :, 1:, :] - image[:, :, :-1, :]
        return gx, gy

    def forward(self, image_pr, image_gt):
        pr_gx, pr_gy = self.gradient(image_pr)
        gt_gx, gt_gy = self.gradient(image_gt)
        loss_gx = F.l1_loss(pr_gx, gt_gx, reduction='mean')
        loss_gy = F.l1_loss(pr_gy, gt_gy, reduction='mean')
        return (loss_gx + loss_gy) / 2.