
from .resnet import *
from .mobilenet_v3 import *


BackboneDict = dict(
    resnet50=resnet.resnet50,
    resnet101=resnet.resnet101,
    wide_resnet50=wide_resnet50,
    resnext101_32x8d=resnext101_32x8d,
    mobilenetv3_small=mobilenet_v3_small,
    mobilenetv3_large=mobilenet_v3_large,
)