from torch import nn
from braincog.base.node.node import *
from functools import partial

class LayerWiseConvModule(nn.Module):
    """
    SNN卷积模块
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: padding
    :param bias: Bias
    :param node: 神经元类型
    :param kwargs:
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 bias=False,
                 node=BiasLIFNode,
                 step=6,
                 **kwargs):

        super().__init__()

        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        self.conv = nn.Conv2d(in_channels=in_channels * self.groups,
                              out_channels=out_channels * self.groups,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias)
        self.gn = nn.GroupNorm(self.groups, out_channels * self.groups)
        self.node = partial(node, **kwargs)()
        self.step = step
        self.activation = nn.Identity()

    def forward(self, x):
        x = rearrange(x, '(t b) c w h -> t b c w h', t=self.step)
        outputs = []
        for t in range(self.step):
            outputs.append(self.gn(self.conv(x[t])))
        outputs = torch.stack(outputs)  # t b c w h
        outputs = rearrange(outputs, 't b c w h -> (t b) c w h')
        outputs = self.node(outputs)
        return outputs


class TEP(nn.Module):
    def __init__(self, step, channel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TEP, self).__init__()
        self.step = step
        self.gn = nn.GroupNorm(channel, channel)

    def forward(self, x):
        x = rearrange(x, '(t b) c w h -> t b c w h', t=self.step)
        fire_rate = torch.mean(x, dim=0)
        fire_rate = self.gn(fire_rate) + 1

        x = x * fire_rate
        x = rearrange(x, 't b c w h -> (t b) c w h')

        return x


class LayerWiseLinearModule(nn.Module):
    """
    线性模块
    :param in_features: 输入尺寸
    :param out_features: 输出尺寸
    :param bias: 是否有Bias, 默认 ``False``
    :param node: 神经元类型, 默认 ``LIFNode``
    :param args:
    :param kwargs:
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 node=BiasLIFNode,
                 step=6,
                 spike=True,
                 *args,
                 **kwargs):
        super().__init__()
        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        if self.groups == 1:
            self.fc = nn.Linear(in_features=in_features,
                                out_features=out_features, bias=bias)
        else:
            self.fc = nn.ModuleList()
            for i in range(self.groups):
                self.fc.append(nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias
                ))
        self.node = partial(node, **kwargs)()
        self.step = step
        self.spike = spike

    def forward(self, x):
        if self.groups == 1:  # (t b) c
            x = rearrange(x, '(t b) c -> t b c', t=self.step)
            outputs = []
            for t in range(self.step):
                outputs.append(self.fc(x[t]))
            outputs = torch.stack(outputs)  # t b c
            outputs = rearrange(outputs, 't b c -> (t b) c')
        else:  # b (c t)
            x = rearrange(x, 'b (c t) -> t b c', t=self.groups)
            outputs = []
            for i in range(self.groups):
                outputs.append(self.fc[i](x[i]))
            outputs = torch.stack(outputs)  # t b c
            outputs = rearrange(outputs, 't b c -> b (c t)')
        if self.spike:
            return self.node(outputs)
        else:
            return outputs