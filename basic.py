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


class LIFNode(BaseNode):
    """
    Leaky Integrate and Fire
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        # self.threshold = threshold
        # print(threshold)
        # print(tau)

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class DoubleSidePLIFNode(LIFNode):
    """
    能够输入正负脉冲的 PLIF
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 tau=2.,
                 act_fun=AtanGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold, tau, act_fun, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)

    def calc_spike(self):

        # self.spike = self.act_fun(self.mem - self.get_thres()) - self.act_fun(self.get_thres - self.mem)
        mem_minus_thres = self.mem - self.get_thres()
        thres_minus_mem = self.get_thres() - self.mem
        self.spike = self.act_fun(mem_minus_thres) - self.act_fun(thres_minus_mem)

        self.mem = self.mem * (1. - torch.abs(self.spike.detach()))
    # def calc_spike(self):
    #     delta_mem = self.mem - self.get_thres()
    #     self.spike = self.act_fun(delta_mem) - self.act_fun(-delta_mem)
    #     self.mem = self.mem * (1. - torch.abs(self.spike.detach()))

