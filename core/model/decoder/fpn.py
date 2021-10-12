
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.transition_in = self._make_transition(in_planes, out_planes)
        self.transition_out = self._make_transition(out_planes, out_planes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _make_transition(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, bot):
        bot = self.transition_in(bot)
        bot = self.upsample(bot)
        return self.transition_out(bot)


class FeaturePyramid(nn.Module):
    def __init__(self, in_planes_list, out_planes):
        super(FeaturePyramid, self).__init__()
        self.in_planes_list = in_planes_list
        self.out_planes = out_planes
        self._make_pyramid()

    def _make_pyramid(self):
        self.transitions_list = list()
        num_skip = len(self.in_planes_list)
        for n in range(num_skip):
            num_transition = num_skip - n
            sequential = list()
            in_planes_list = [self.in_planes_list[n] // (2**i) for i in range(num_transition)]
            out_planes_list = in_planes_list[1:] + [self.out_planes]
            for in_planes, out_planes in zip(in_planes_list, out_planes_list):
                sequential.append(Transition(in_planes, out_planes))
            setattr(self, 'transition{}'.format(num_skip+1-n), nn.Sequential(*sequential))

    def forward(self, skip_list):
        num_skip = len(self.in_planes_list)
        assert len(skip_list) == num_skip
        body_list = []
        for n in range(len(skip_list)):
            transition = getattr(self, 'transition{}'.format(num_skip + 1 - n))
            body = transition(skip_list[n])
            body_list.append(body)
        return sum(body_list)