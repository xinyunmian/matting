
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        bot = F.interpolate(bot, size=skip.shape[2:], mode='bilinear', align_corners=True)
        skip = self.skip(skip)
        cate = torch.cat((bot, skip), dim=1)
        fuse = self.fuse(cate)
        return fuse


class MattingHead(nn.Module):
    def __init__(self, in_planes, skip_planes):
        super(MattingHead, self).__init__()
        self.in_planes = in_planes
        self.skip_planes = skip_planes
        self.skip_fuse3 = FusionSkip(skip_planes[3], in_planes // 1, in_planes // 2)
        self.skip_fuse2 = FusionSkip(skip_planes[2], in_planes // 2, in_planes // 4)
        self.skip_fuse1 = FusionSkip(skip_planes[1], in_planes // 4, in_planes // 8)
        self.skip_fuse0 = FusionSkip(skip_planes[0], in_planes // 8, in_planes // 16)
        self.post = nn.Conv2d(in_planes // 16, 1, 5, 1, 2, 1, bias=False)

    def forward(self, skip_list, body):
        c0, c1, c2, c3 = skip_list
        body = self.skip_fuse3(body, c3)
        body = self.skip_fuse2(body, c2)
        body = self.skip_fuse1(body, c1)
        body = self.skip_fuse0(body, c0)
        post = self.post(body)
        output = torch.clamp(post, 0., 1.)
        return output


class MattingHeadC(nn.Module):
    def __init__(self, in_planes, skip_planes):
        super(MattingHeadC, self).__init__()
        self.in_planes = in_planes
        self.skip_planes = skip_planes
        self._make_head()

    def _make_head(self):
        num_skip = len(self.skip_planes)
        in_plane_list = [self.in_planes // (2 ** n) for n in range(0, num_skip)]
        out_plane_list = [self.in_planes // (2 ** n) for n in range(1, num_skip + 1)]
        for n in range(num_skip):
            skip_planes = self.skip_planes[num_skip - 1 - n]
            setattr(self, 'skip_fuse{}'.format(num_skip - n - 1),
                    FusionSkip(skip_planes, in_plane_list[n], out_plane_list[n]))
        self.post = nn.Conv2d(out_plane_list[-1], 1, 1, 1, 0, 1, bias=False)

    def forward(self, x, skip_list, body):
        num_skip = len(self.skip_planes)
        assert len(skip_list) == num_skip
        for n in range(len(skip_list)):
            fuse = getattr(self, 'skip_fuse{}'.format(num_skip - n - 1))
            body = fuse(body, skip_list[num_skip - 1 - n])
        post = self.post(body)
        output = torch.clamp(post, 0., 1.)
        return output



class ProgressiveRefineMattingHead(nn.Module):
    def __init__(self, in_planes, skip_planes):
        super(ProgressiveRefineMattingHead, self).__init__()
        self.skip_fuse3 = FusionSkip(skip_planes[3], in_planes // 1, in_planes // 2)
        self.skip_fuse2 = FusionSkip(skip_planes[2], in_planes // 2, in_planes // 4)
        self.skip_fuse1 = FusionSkip(skip_planes[1], in_planes // 4, in_planes // 8)
        self.skip_fuse0 = FusionSkip(skip_planes[0], in_planes // 8, in_planes // 16)
        self.fuse3_head = nn.Conv2d(in_planes // 2, 1, 5, 1, 2, 1, bias=False)
        self.fuse2_head = nn.Conv2d(in_planes // 4, 1, 5, 1, 2, 1, bias=False)
        self.fuse1_head = nn.Conv2d(in_planes // 8, 1, 5, 1, 2, 1, bias=False)
        self.fuse0_head = nn.Conv2d(in_planes // 16, 1, 5, 1, 2, 1, bias=False)

    def _progressive_refine(self, pred, bot):
        up = F.interpolate(bot, scale_factor=2, mode='bilinear', align_corners=True)
        clip = torch.clamp(up, 0., 1.)
        mask0 = torch.where(0. < clip, 1, 0)
        mask1 = torch.where(clip < 1., 1, 0)
        tri = mask0 * mask1
        mat = pred * tri + (1 - tri) * up
        return torch.clamp(mat, 0., 1.)

    def forward(self, skip_list, body):
        c0, c1, c2, c3 = skip_list
        fuse3 = self.skip_fuse3(body, c3)
        pred3 = self.fuse3_head(fuse3)

        fuse2 = self.skip_fuse2(fuse3, c2)
        pred2 = self.fuse2_head(fuse2)
        pred2 = self._progressive_refine(pred2, pred3)

        fuse1 = self.skip_fuse1(fuse2, c1)
        pred1 = self.fuse1_head(fuse1)
        pred1 = self._progressive_refine(pred1, pred2)

        fuse0 = self.skip_fuse0(fuse1, c0)
        pred0 = self.fuse0_head(fuse0)
        pred0 = self._progressive_refine(pred0, pred1)

        return [pred0, pred1, pred2]
