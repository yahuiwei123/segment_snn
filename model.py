import torch

from fpn import FPNSegmentationHead
from vgg16 import VGG16
from torch import nn
from braincog.base.node.node import *


class SegmentModel(nn.Module):
    def __init__(self, output_size, out_cls, node=BiasLIFNode, step=6):
        super(SegmentModel, self).__init__()
        self.output_size = output_size
        self.node = node
        self.step = step
        self.encoder = VGG16(out_cls=out_cls, step=step)
        self.decoder = FPNSegmentationHead(512, out_cls,
                                           decode_intermediate_input=True,
                                           shortcut_dims=[64, 128, 256, 512],
                                           node=node,
                                           step=step,
                                           align_corners=True)

    def forward(self, x: torch.Tensor = None):
        """
        x -> t b c w h
        """
        embs = self.encoder(x)
        for i in range(len(embs)):
            embs[i] = rearrange(embs[i], '(t b) c w h -> t b c w h', t=self.step)
        logits = self.decoder(embs[-1], embs[0:-1])
        logits = rearrange(logits, '(t b) c w h -> t b c w h', t=self.step)
        out_logits = torch.mean(logits, dim=0)
        out_logits = F.interpolate(out_logits,
                                   size=self.output_size,
                                   mode="bilinear",
                                   align_corners=True)
        return out_logits
