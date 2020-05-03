import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from utils.downsample import Downsample
from pushpull.PPmodule2d import PPmodule2d

"""
ResNet with Push-Pull: implemented on top of the official PyTorch ResNet implementation
"""

__all__ = ['ResNetCifar', 'resnet20', 'resnet32', 'resnet44', 'resnet56']

model_urls = {
    'resnet20': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet20-pp': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet32': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet32-pp': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet44': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet44-pp': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet56': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet56-pp': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size_lpf=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes)
        else:
            if size_lpf is None:
                self.conv1 = conv3x3(inplanes, planes, stride=stride)
            else:
                self.conv1 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=inplanes),
                                       conv3x3(inplanes, planes), )

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, size_lpf=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, bias=False)
        else:
            if size_lpf is None:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                       padding=1, bias=False, stride=stride)
            else:
                self.conv2 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=planes),
                                           nn.Conv2d(planes, planes, kernel_size=3,
                                                     padding=1, bias=False),)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PushPullBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, train_alpha=False, size_lpf=None):
        super(PushPullBlock, self).__init__()
        if stride == 1:
            self.pp1 = PPmodule2d(inplanes, planes, kernel_size=3, padding=1, bias=False,
                                  # alpha=alpha_pp, scale=scale_pp,
                                  train_alpha=train_alpha)
        else:
            if size_lpf is None:
                self.pp1 = PPmodule2d(inplanes, planes, kernel_size=3, padding=1, bias=False,
                                      # alpha=alpha_pp, scale=scale_pp,
                                      train_alpha=train_alpha, stride=stride)
            else:
                self.pp1 = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=inplanes),
                                     PPmodule2d(inplanes, planes, kernel_size=3,
                                                padding=1, bias=False, train_alpha=train_alpha), )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pp2 = PPmodule2d(planes, planes, kernel_size=3, 
                              padding=1, bias=False,  # alpha=alpha_pp, scale=scale_pp,
                              train_alpha=train_alpha)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.pp1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pp2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetCifar(nn.Module):
    """
    ResNet with Push-Pull for CIFAR: implemented on top of the official PyTorch ResNet implementation

    args:
        use_pp1 (bool, optional): if ''True'', use the Push-Pull layer to replace the first conv layer (default: False)
        pp_all (bool, optional): if ''True'', use the Push-Pull layer to replace all conv layers (default: False)
        pp_block1 (bool, optional): if ''True'', use the Push-Pull layer to replace all conv layers in the 1st residual block (default: False)
        train_alpha (bool, optional): if ''True'', the inhibition strength 'alpha' is trainable (default: False)
        size_lpf (int, optional): if specified, it uses an LPF filter of size ('size_lpf' x 'size_lpf') before downsampling operation (Zhang's paper) (default: None)
    """
    def __init__(self, block, layers, num_classes=10,
                 use_pp1=False, pp_all=False,
                 pp_block1=False, train_alpha=False, size_lpf=None):

        self.inplanes = 16
        super(ResNetCifar, self).__init__()

        if use_pp1:
            self.conv1 = PPmodule2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, train_alpha=train_alpha)
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        if pp_all:
            # Use push-pull inhibition at all layers
            self.layer1 = self._make_layer(PushPullBlock, 16, layers[0], train_alpha=train_alpha)
            self.layer2 = self._make_layer(PushPullBlock, 32, layers[1], train_alpha=train_alpha,
                                           stride=2, size_lpf=size_lpf)
            self.layer3 = self._make_layer(PushPullBlock, 64, layers[2], train_alpha=train_alpha,
                                           stride=2, size_lpf=size_lpf)
        else:
            # use push-pull inhibition in the first residual block only
            if pp_block1:
                self.layer1 = self._make_layer(PushPullBlock, 16, layers[0], train_alpha=train_alpha)
            else:
                self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2, size_lpf=size_lpf)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, size_lpf=size_lpf)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, train_alpha=False, size_lpf=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if size_lpf is None:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                # downsample according to Nyquist (from the paper of Zhang)
                downsample = nn.Sequential(Downsample(filt_size=size_lpf, stride=stride, channels=self.inplanes),
                                           nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                                           nn.BatchNorm2d(planes * block.expansion)
                                           )

        layers = []
        if block is PushPullBlock:
            layers.append(block(self.inplanes, planes, stride, downsample, train_alpha=train_alpha, size_lpf=size_lpf))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, size_lpf=size_lpf))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if block is PushPullBlock:
                layers.append(block(self.inplanes, planes, train_alpha=train_alpha, size_lpf=size_lpf))
            else:
                layers.append(block(self.inplanes, planes, size_lpf=size_lpf))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model

