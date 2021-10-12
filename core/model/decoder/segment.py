
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn import FeaturePyramid


class FusionSkip(nn.Module):
    def __init__(self, skip_planes, bot_planes, out_planes):
        super(FusionSkip, self).__init__()
        self.skip_planes = skip_planes
        self.bot_planes = bot_planes
        self.out_planes = out_planes
        self.skip = self._make_skip(skip_planes, out_planes)
        self.fuse = self._make_fusion(bot_planes, out_planes)

    def _make_skip(self, skip_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(skip_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def _make_fusion(self, bot_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(bot_planes+out_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, bot, skip):
        bot = F.interpolate(bot, size=skip.size()[2:], mode='bilinear', align_corners=True)
        skip = self.skip(skip)
        cate = torch.cat((bot, skip), dim=1)
        fuse = self.fuse(cate)
        return fuse


class FPNHead(nn.Module):
    def __init__(self, in_planes, out_planes, skip_planes):
        super(FPNHead, self).__init__()
        self.skip_fuse4 = FusionSkip(skip_planes[2], in_planes // 1, in_planes // 2)
        self.skip_fuse3 = FusionSkip(skip_planes[1], in_planes // 2, in_planes // 4)
        self.skip_fuse2 = FusionSkip(skip_planes[0], in_planes // 4, in_planes // 8)
        self.fpn = FeaturePyramid([in_planes // 2, in_planes // 4, in_planes // 8], in_planes // 16)
        self.post = nn.Conv2d(in_planes // 16, out_planes, 1, 1, 0, 1, bias=False)

    def forward(self, x, skip_list, body):
        assert len(skip_list) == 3
        c1, c2, c3 = skip_list
        body4 = self.skip_fuse4(body, c3)
        body3 = self.skip_fuse3(body4, c2)
        body2 = self.skip_fuse2(body3, c1)
        post = self.post(self.fpn(body2, body3, body4))
        output = F.interpolate(post, size=x.size()[2:], mode='bilinear', align_corners=True)
        return output



class FPNHeadC(nn.Module):
    def __init__(self, in_planes, out_planes, skip_planes, **kwargs):
        super(FPNHeadC, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.skip_planes = skip_planes
        self._make_head()

    def _make_head(self):
        num_skip = len(self.skip_planes)
        fpn_out_planes = self.in_planes // (2**(num_skip+1))
        fpn_in_plane_list = [self.in_planes // (2**n) for n in range(0, num_skip)]
        fpn_out_plane_list = [self.in_planes // (2**n) for n in range(1, num_skip+1)]
        for n in range(num_skip):
            skip_planes = self.skip_planes[num_skip-1-n]
            setattr(self, 'skip_fuse{}'.format(num_skip + 1 - n),
                    FusionSkip(skip_planes, fpn_in_plane_list[n], fpn_out_plane_list[n]))
        setattr(self, 'fpn', FeaturePyramid(fpn_out_plane_list, fpn_out_planes))
        self.post = nn.Conv2d(fpn_out_planes, self.out_planes, 1, 1, 0, 1, bias=False)

    def forward(self, x, skip_list, body):
        num_skip = len(self.skip_planes)
        assert len(skip_list) == num_skip
        body_list = []
        for n in range(len(skip_list)):
            fuse = getattr(self, 'skip_fuse{}'.format(num_skip + 1 - n))
            body = fuse(body, skip_list[num_skip-1-n])
            body_list.append(body)
        post = self.post(self.fpn(body_list))
        output = F.interpolate(post, size=x.size()[2:], mode='bilinear', align_corners=True)
        return output



class DeepLabHead(nn.Module):
    def __init__(self, c1_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(c1_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, bot, c1):
        bot = F.interpolate(bot, c1.size()[2:], mode='bilinear', align_corners=True)
        skip = self.skip(c1)
        bot = self.head(torch.cat([bot, skip], dim=1))
        bot = F.interpolate(bot, size=x.size()[2:], mode='bilinear', align_corners=True)
        return bot