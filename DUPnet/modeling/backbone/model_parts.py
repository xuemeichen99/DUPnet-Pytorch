import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from typing import Any, List, Tuple
from collections import OrderedDict
from modeling.aspp import build_mspp

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


#下采样卷积
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm = nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x



class _DenseLayer(nn.Module):
    """DenseBlock中的内部结构 DenseLayer: BN + ReLU + Conv(1x1) + BN + ReLU + Conv(3x3)"""

    def __init__(self,
                 num_input_features: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        """
        :param num_input_features: 输入channel
        :param growth_rate: 论文中的 k = 32
        :param bn_size: 1x1卷积的filternum = bn_size * k  通常bn_size=4
        :param drop_rate: dropout 失活率
        :param memory_efficient: Memory-efficient版的densenet  默认是不使用的
        """
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=False))
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=False))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient


    def bn_function(self, inputs: List[Tensor]) -> Tensor:


        concat_features = torch.cat(inputs, 1)

        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        """判断是否需要更新梯度（training）"""
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        """
        torch.utils.checkpoint: 用计算换内存（节省内存）。 详情可看： https://arxiv.org/abs/1707.06990
        torch.utils.checkpoint并不保存中间激活值，而是在反向传播时重新计算它们。 它可以应用于模型的任何部分。
        具体而言，在前向传递中,function将以torch.no_grad()的方式运行,即不存储中间激活值 相反,前向传递将保存输入元组和function参数。
        在反向传播时，检索保存的输入和function参数，然后再次对函数进行正向计算，现在跟踪中间激活值，然后使用这些激活值计算梯度。
        """

        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs


        if self.memory_efficient and self.any_requires_grad(prev_features):

            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:

            bottleneck_output = self.bn_function(prev_features)


        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        """
        :param num_layers: 该DenseBlock中DenseLayer的个数
        :param num_input_features: 该DenseBlock的输入Channel，每经过一个DenseBlock都会进行叠加
                        叠加方式：num_features = num_features + num_layers * growth_rate
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param growth_rate: 指的是论文中的k  小点比较好  论文中是32
        :param drop_rate: dropout rate after each dense layer
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:

        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, 1)

class Down(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 num_output_features:int,
                 exit_block_dilations=(1, 2),
                 BatchNorm = nn.BatchNorm2d):
        super(Down, self).__init__()

        self.bn = BatchNorm(num_input_features)
        self.relu = nn.ReLU(inplace=False)
        self.down_layer = SeparableConv2d(num_input_features, num_output_features, 3, stride=2, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)


    def forward(self,x):
        y1 = self.bn(x)
        y2 = self.relu(y1)
        y3 = self.down_layer(y2)

        return y3


class Up(nn.Sequential):
    def __init__(self,num_input_features: int,
                 num_output_features: int):
        super(Up, self).__init__()
        self.upsample_layer = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(),
            nn.Conv2d(num_input_features,num_output_features,kernel_size = 1,stride = 1,bias = False)

            )
        self.conv1 = nn.Conv2d(num_output_features*2,num_output_features,kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=False)



    def forward(self,x,y):
        x1 = self.upsample_layer(x)
        y1= F.interpolate(y, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat([x1,y1],dim = 1)
        x5 = self.relu(self.bn(self.conv1(x3)))
        return x5


class Dense_4(nn.Module):
    """Densenet-BC"""

    def __init__(self,
                 growth_rate: int = 24,
                 block_config: Tuple[int, int] = (4, 0),
                 num_init_features: int=128,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 2,
                 memory_efficient: bool = False):
        """
        :param growth_rate: 指的是论文中的k  小点比较好  论文中是32
        :param block_config: 每一个DenseBlock中_DenseLayer的个数
        :param num_init_features: 整个网络第一个卷积（Conv0）的kernel_size = 64
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param drop_rate: dropout rate after each dense layer 一般为0 不用的
        :param num_classes: 数据集类别数
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient
        """
        super(Dense_4, self).__init__()


        self.features = nn.Sequential(OrderedDict([
            ("relu0", nn.ReLU(inplace=False))
        ]))

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate



    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)




class Dense_Unet(nn.Module):
    def __init__(self, out_chan = 2):
        super(Dense_Unet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, 64, kernel_size=3,padding=1, stride=1, bias=False)),
            ("norm0", nn.BatchNorm2d(64)),
            ("relu0", nn.ReLU()),
            ("spre",  nn.Conv2d(64, 64, 3,padding=1)),
            ("norm2", nn.BatchNorm2d(64)),
            ("relu3", nn.ReLU())

        ]))
        self.d0 = Dense_4(num_init_features=64, block_config=(4, 0))
        self.down1 = Down(160, 256)
        self.d1 = Dense_4(num_init_features=256, block_config=(4, 0))
        self.down2 = Down(352, 512)
        self.d2 = Dense_4(num_init_features=512, block_config=(4, 0))
        self.down3 = Down(608, 1024)
        self.d3 = Dense_4(num_init_features=1024, block_config=(4, 0))
        self.down4 = Down(1120, 1120)
        self.d4 = Dense_4(num_init_features=1120, block_config=(4, 0))


        self.up4 = Up(1216,608)
        self.u4 = Dense_4(num_init_features=608, block_config=(4, 0))
        self.up3 = Up(704,352)
        self.u3 = Dense_4(num_init_features=352, block_config=(4, 0))
        self.up2 = Up(448,224)
        self.u2 = Dense_4(num_init_features=224, block_config=(4, 0))
        self.up1 = Up(320,160)
        self.u1 = Dense_4(num_init_features=160, block_config=(4, 0))
        self.last_conv = nn.Sequential(
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(128, out_chan, kernel_size=1, stride=1))

        self.aspp1 = build_mspp(inplanes=160, output_stride=16,upout=160)
        self.aspp2 = build_mspp(inplanes=352, output_stride=16,upout=224)
        self.aspp3 = build_mspp(inplanes=608, output_stride=16,upout=352)
        self.aspp4 = build_mspp(inplanes=1120, output_stride=16,upout=608)

        self._init_weight()

    def forward(self, x_in):
        x0 = self.features(x_in)
        xd0 = self.d0(x0)
        y11 = self.aspp1(xd0)
        y1 = self.d1(self.down1(xd0))
        y22 = self.aspp2(y1)
        y2 = self.d2(self.down2(y1))
        y33 = self.aspp3(y2)
        y3 = self.d3(self.down3(y2))
        y44 = self.aspp4(y3)
        y4 = self.d4(self.down4(y3))
        x6 = self.u4(self.up4(y4, y44))
        x7 = self.u3(self.up3(x6, y33))
        x8 = self.u2(self.up2(x7, y22))
        x9 = self.u1(self.up1(x8, y11))
        x11 = self.last_conv(x9)
        return x11

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    """测试模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Dense_Unet(out_chan = 2)
    print(model)
    model.eval()
    input = torch.rand(3, 3,128,128)
    output = model(input)
    print(output.shape)
    print(output)

