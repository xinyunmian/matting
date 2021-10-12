
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256):
        super(ASPP, self).__init__()
        self.atrous_rates = atrous_rates
        self.out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()))

        # rate1, rate2, rate3 = tuple(atrous_rates)
        # modules.append(ASPPConv(in_channels, self.out_channels, rate1))
        # modules.append(ASPPConv(in_channels, self.out_channels, rate2))
        # modules.append(ASPPConv(in_channels, self.out_channels, rate3))

        modules.extend(self._make_asppconv(in_channels))
        modules.append(ASPPPooling(in_channels, self.out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

    def _make_asppconv(self, in_channels):
        modules = []
        for dilate in self.atrous_rates:
            modules.append(ASPPConv(in_channels, self.out_channels, dilate))
        return modules

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


