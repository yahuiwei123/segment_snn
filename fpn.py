import torch
import torch.nn as nn
import torch.nn.functional as F
from basic import *

class FPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[64, 128, 256, 512],
                 node=BiasLIFNode,
                 step=6,
                 align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.node = node
        self.step = step
        self.decode_intermediate_input = decode_intermediate_input


        # self.conv_in = ConvGN(in_dim, hidden_dim, 1)
        self.conv_in = LayerWiseConvModule(in_dim, hidden_dim, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)

        # self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        # self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        # self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)
        self.conv_16x = LayerWiseConvModule(hidden_dim, hidden_dim, 3, node=BiasLIFNode, step=self.step)
        self.conv_8x = LayerWiseConvModule(hidden_dim, hidden_dim // 2, 3, node=BiasLIFNode, step=self.step)
        self.conv_4x = LayerWiseConvModule(hidden_dim // 2, hidden_dim // 2, 3, node=BiasLIFNode, step=self.step)
        self.conv_2x = LayerWiseConvModule(hidden_dim // 2, hidden_dim // 2, 3, node=BiasLIFNode, step=self.step)
        # self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        # self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
        # self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)
        self.in_tep = TEP(step=self.step, channel=hidden_dim, device=None, dtype=None)
        self.adapter_16x = LayerWiseConvModule(shortcut_dims[-1], hidden_dim, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)
        self.tep_16x = TEP(step=self.step, channel=hidden_dim, device=None, dtype=None)
        self.adapter_8x = LayerWiseConvModule(shortcut_dims[-2], hidden_dim, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)
        self.tep_8x = TEP(step=self.step, channel=hidden_dim // 2, device=None, dtype=None)
        self.adapter_4x = LayerWiseConvModule(shortcut_dims[-3], hidden_dim // 2, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)
        self.tep_4x = TEP(step=self.step, channel=hidden_dim // 2, device=None, dtype=None)
        self.adapter_2x = LayerWiseConvModule(shortcut_dims[-4], hidden_dim // 2, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)
        self.tep_2x = TEP(step=self.step, channel=hidden_dim // 2, device=None, dtype=None)

        self.conv_out = LayerWiseConvModule(hidden_dim // 2, out_dim, 1, padding=(0, 0), node=BiasLIFNode, step=self.step)

        self._init_weight()

    def forward(self, inputs, shortcuts):
        self.reset()

        inputs = rearrange(inputs, 't b c w h -> (t b) c w h')
        # print("fph-input.shape")
        # print(inputs.shape)
        for i in range(len(shortcuts)):
            shortcuts[i] = rearrange(shortcuts[i], 't b c w h -> (t b) c w h')

        # if self.decode_intermediate_input:
        #     x = torch.cat(inputs, dim=1)
        # else:
        #     x = inputs[-1]
        if self.decode_intermediate_input:
            x = self.in_tep(self.conv_in(inputs))
        else:
            x = inputs

        x = F.interpolate(x,
                          size=shortcuts[-1].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = self.tep_16x(self.conv_16x(self.adapter_16x(shortcuts[-1]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-2].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = self.tep_8x(self.conv_8x(self.adapter_8x(shortcuts[-2]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = self.tep_4x(self.conv_4x(self.adapter_4x(shortcuts[-3]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = self.tep_2x(self.conv_2x(self.adapter_2x(shortcuts[-4]) + x))

        x = self.conv_out(x)
        # print("fpn-output")
        # print(x.shape)

        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset(self):
        """
        重置所有神经元的膜电位
        :return:
        """
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
