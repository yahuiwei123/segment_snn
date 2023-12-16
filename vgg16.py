from torch import nn
from basic import *

"""
  平均池化后做差
   kernel_size:平均池化窗口大小
   """
kernel_size = 2
class CustomAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(CustomAvgPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        pooled = self.avgpool(x)
        upsampled = F.interpolate(pooled, scale_factor=kernel_size, mode='nearest')
        diff = x - upsampled
        return diff

class VGG16(nn.Module):
    def __init__(self, node=DoubleSidePLIFNode, out_cls=10, step=6, **kwargs):  # 1   3e38
        super(VGG16, self).__init__()
        self.node = node
        self.step = step
        self.custom_avg_pool = CustomAvgPool2d(kernel_size=2, stride=2, padding=0)
        self.downsample_2x = nn.Sequential(
            LayerWiseConvModule(2, 64, 3, 1, 1, node=BiasLIFNode, step=self.step),
            TEP(step=self.step, channel=64, device=None, dtype=None),
            LayerWiseConvModule(64, 64, 3, 1, 1, node=BiasLIFNode, step=self.step),
            TEP(step=self.step, channel=64, device=None, dtype=None),
            self.custom_avg_pool,
            nn.MaxPool2d(2, 2)
        )

        self.downsample_4x = nn.Sequential(
            LayerWiseConvModule(64, 128, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=128, device=None, dtype=None),
            LayerWiseConvModule(128, 128, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=128, device=None, dtype=None),
            nn.MaxPool2d(2, 2)
        )

        self.downsample_8x = nn.Sequential(
            LayerWiseConvModule(128, 256, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=256, device=None, dtype=None),
            LayerWiseConvModule(256, 256, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=256, device=None, dtype=None),
            nn.MaxPool2d(2, 2)
        )

        self.downsample_16x = nn.Sequential(
            LayerWiseConvModule(256, 512, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=512, device=None, dtype=None),
            LayerWiseConvModule(512, 512, 3, 1, 1, node=DoubleSidePLIFNode, step=self.step),
            TEP(step=self.step, channel=512, device=None, dtype=None),
            nn.MaxPool2d(2, 2)
        )

        self.downsample_32x = nn.Sequential(
            LayerWiseConvModule(512, 512, 3, 1, 1, node, step=self.step),
            TEP(step=self.step, channel=512, device=None, dtype=None),
            LayerWiseConvModule(512, 512, 3, 1, 1, node, step=self.step),
            TEP(step=self.step, channel=512, device=None, dtype=None),
            nn.MaxPool2d(2, 2)
        )

        self.fc = LayerWiseLinearModule(512, out_cls, bias=True, node=DoubleSidePLIFNode, step=self.step)
        # self.node = partial(node, **kwargs)()

    def forward(self, input):
        self.reset()
        # input = input.permute(1, 0, 2, 3, 4)
        input = rearrange(input, 't b c w h -> (t b) c w h')

        # embedding
        downsample_2x = self.downsample_2x(input)
        downsample_4x = self.downsample_4x(downsample_2x)
        downsample_8x = self.downsample_8x(downsample_4x)
        downsample_16x = self.downsample_16x(downsample_8x)
        downsample_32x = self.downsample_32x(downsample_16x)

        shortcuts = [downsample_2x, downsample_4x, downsample_8x, downsample_16x, downsample_32x]

        # x = downsample_32x.view(downsample_32x.shape[0], -1)
        # output = self.fc(x)
        # outputs = rearrange(output, '(t b) c -> t b c', t=self.step)

        return shortcuts  # sum(outputs) / len(outputs)

    def reset(self):
        """
        重置所有神经元的膜电位
        :return:
        """
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
