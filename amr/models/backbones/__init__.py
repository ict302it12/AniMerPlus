from .vit_moe import vithmoe
from torch import nn
import torchvision

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vithmoe':
        return vithmoe(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
