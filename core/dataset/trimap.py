
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import skimage.measure as measure
from scipy import signal


class TrimapGeneration(nn.Module):
    def __init__(self):
        super(TrimapGeneration, self).__init__()
        self.near = 5

    def forward(self, matting):
        with torch.no_grad():
            matting = torch.unsqueeze(matting, dim=1)
            matting_bdr = torch.where((matting > 0) & (matting < 1), torch.ones_like(matting), torch.zeros_like(matting)).float()
            kernel = torch.ones((1, 1, self.near, self.near)).float()
            matting_bdr = F.conv2d(matting_bdr, kernel, bias=None, stride=1, padding=self.near // 2, dilation=1)
            matting_bdr = torch.where(matting_bdr > 0, torch.ones_like(matting_bdr), torch.zeros_like(matting_bdr)).long()
            output = torch.zeros_like(matting_bdr)
            output[matting == 1] = 2
            output[matting_bdr == 1] = 1
            output = torch.squeeze(output, dim=1)
            return output


class TrimapAdaptive(nn.Module):
    def __init__(self):
        super(TrimapAdaptive, self).__init__()

    def forward(self, tri_pr, tri_gt):
        # extend the tri-map region
        with torch.no_grad():
            N, _, H, W = tri_pr.shape
            tri_gt = tri_gt.view(N, 1, H, W)
            transition = torch.where(tri_gt == 1, torch.ones_like(tri_gt), torch.zeros_like(tri_gt)).float()
            kernel = torch.ones((1, 1, 17, 17)).float().cuda()
            output = F.conv2d(transition, kernel, bias=None, stride=1, padding=8, dilation=1)
            tri_ext = torch.where(output > 0, torch.ones_like(output), torch.zeros_like(output))
            tri_ext = tri_ext.view(N, H, W)
            arg_gt = tri_gt.long()
            arg_pr = torch.argmax(torch.reshape(tri_pr, (-1, 3)), dim=1, keepdim=False).long()
            arg_gt = torch.reshape(arg_gt, (N, H, W))
            arg_pr = torch.reshape(arg_pr, (N, H, W))
            mask = (arg_pr == 1).long() * (arg_gt != 1).long() * (tri_ext == 1).long()
            return (torch.ones_like(mask) - mask).float().view(N, H, W)


class TrimapGenerator:
    def __init__(self):
        self.filter_tri, self.flag_tri = self.calculate_filter(k=11)
        self.filter_pre, self.flag_pre = self.calculate_filter(k=3)

    def calculate_filter(self, k: int = 11):
        cx, cy = k // 2, k // 2
        center = np.reshape(np.array([cx, cy], dtype=np.int32), (1, 1, 2))
        xx = np.arange(0, k, 1)
        yy = np.arange(0, k, 1)
        x, y = np.meshgrid(xx, yy)
        mesh = np.stack([x, y], axis=2).astype(np.int32)
        ds = np.square(mesh - center)
        distance = np.sum(ds, axis=2)
        filter = np.where(distance < 150, 1, 0).astype(np.int32)
        return filter, np.sum(filter)

    def calculate_tri(self, mask, filter, flag):
        bdr = signal.convolve2d(mask, filter, mode='same', boundary='symm')
        bdr = np.where((bdr > 0) & (bdr < flag), 1, 0)
        output = np.zeros_like(mask)  # background is 0
        output[mask == 1] = 2
        output[bdr == 1] = 1
        return output

    def connectivity_process(self, tri):
        assert len(np.unique(tri) == 3)
        mask = np.where(tri > 0, np.ones_like(tri), np.zeros_like(tri))
        region, n = measure.label(mask, connectivity=2, return_num=True)
        props = measure.regionprops(region)
        num_pix = []
        for ia in range(len(props)):
            num_pix += [props[ia].area]
        max_num = max(num_pix)
        index = num_pix.index(max_num)
        label = props[index].label
        return np.where(region == label, np.ones_like(region), np.zeros_like(region)).astype(np.uint8)

    def calculate_tri_from_seg(self, seg):
        bdr_pre = self.calculate_tri(seg, self.filter_pre, self.flag_pre)
        new_seg = self.connectivity_process(bdr_pre) * seg
        bdr_out = self.calculate_tri(new_seg, self.filter_tri, self.flag_tri)
        return bdr_out

    def __call__(self, matting):
        segment = np.where(matting > 0, np.ones_like(matting), np.zeros_like(matting))
        return self.calculate_tri_from_seg(segment)