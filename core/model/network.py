import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.aspp import ASPP
from core.model.backbone import *
from core.model.decoder import *


class JointMattingParsing(nn.Module):
    def __init__(self, num_classes, backbone):
        super(JointMattingParsing, self).__init__()
        assert backbone in BackboneDict
        self.backbone = BackboneDict[backbone](**self.config_backbone(backbone))
        self.aspp = ASPP(in_channels=self.backbone.bot_channel, atrous_rates=(2, 4, 6), out_channels=256)
        self.seg_head = MultiHeadModule(FPNHeadC, 3, in_planes=self.aspp.out_channels, out_planes=num_classes, skip_planes=self.backbone.skip_channel[1:])
        self.mat_head = MattingHeadC(in_planes=self.aspp.out_channels, skip_planes=self.backbone.skip_channel)

    def config_backbone(self, backbone):
        if 'resnet' in backbone:
            return dict(replace_stride_with_dilation=[False, False, False, True])
        return dict()

    def load_backbone_state_dict(self, state_dict_file):
        self.backbone.load_state_dict_without_fc(state_dict_file, strict=True)

    def config_version(self):
        return torch.tensor([ord(c) for c in 'elves-06']).long()

    def forward(self, x):
        c0, c1, c2, c3, c4, c5 = self.backbone(x)  # c0, c1, c2, c3, c4, c5
        bot = self.aspp(c5)
        seg = self.seg_head(x=x, skip_list=(c1, c2, c3), body=bot)
        mat = self.mat_head(x=x, skip_list=(c0, c1, c2, c3), body=bot)
        return dict(segment=seg, matting=mat)





if __name__ == '__main__':
    net = JointHumanMattingParsing(num_classes=3, backbone='resnet50')
    output = net(torch.ones(2, 3, 256, 256))
    # net.load_backbone_state_dict('M:/pretrain/pytorch/resnet50-19c8e357.pth')

