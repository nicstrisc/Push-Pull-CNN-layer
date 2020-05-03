import torch
import math
from torch import nn
import torch.nn.functional as F


class PPmodule2d(nn.Module):
    """
    Implementation of the Push-Pull layer from:
    [1] N. Strisciuglio, M. Lopez-Antequera, N. Petkov,
    Enhanced robustness of convolutional networks with a push–pull inhibition layer,
    Neural Computing and Applications, 2020, doi: 10.1007/s00521-020-04751-8

    It is an extension of the Conv2d module, with extra arguments:

    * :attr:`alpha` controls the weight of the inhibition. (default: 1 - same strength as the push kernel)
    * :attr:`scale` controls the size of the pull (inhibition) kernel (default: 2 - double size).
    * :attr:`dual_output` determines if the response maps are separated for push and pull components.
    * :attr:`train_alpha` controls if the inhibition strength :attr:`alpha` is trained (default: False).


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        alpha (float, optional): Strength of the inhibitory (pull) response. Default: 1
        scale (float, optional): size factor of the pull (inhibition) kernel with respect to the pull kernel. Default: 2
        dual_output (bool, optional): If ``True``, push and pull response maps are places into separate channels of the output. Default: ``False``
        train_alpha (bool, optional): If ``True``, set alpha (inhibition strength) as a learnable parameters. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 alpha=1, scale=2, dual_output=False,
                 train_alpha=False):
        super(PPmodule2d, self).__init__()

        self.dual_output = dual_output
        self.train_alpha = train_alpha

        # Note: the dual output is not tested yet
        if self.dual_output:
            assert (out_channels % 2 == 0)
            out_channels = out_channels // 2

        # Push kernels (is the one for which the weights are learned - the pull kernel is derived from it)
        self.push = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        """
        # Bias: push and pull convolutions will have bias=0.
        # If the PP kernel has bias, it is computed next to the combination of the 2 convolutions
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            # Inizialize bias
            n = in_channels
            for k in self.push.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        """

        # Configuration of the Push-Pull inhibition
        if not self.train_alpha:
            # when alpha is an hyper-parameter (as in [1])
            self.alpha = alpha
        else:
            # when alpha is a trainable parameter
            k = 1
            self.alpha = nn.Parameter(k * torch.ones(1, out_channels, 1, 1), requires_grad=True)
            r = 1. / math.sqrt(in_channels * out_channels)
            self.alpha.data.uniform_(.5-r, .5+r)  # math.sqrt(n) / 2)  # (-stdv, stdv)

        self.scale_factor = scale
        push_size = self.push.weight[0].size()[1]

        # compute the size of the pull kernel
        if self.scale_factor == 1:
            pull_size = push_size
        else:
            pull_size = math.floor(push_size * self.scale_factor)
            if pull_size % 2 == 0:
                pull_size += 1

        # upsample the pull kernel from the push kernel
        self.pull_padding = pull_size // 2 - push_size // 2 + padding
        self.up_sampler = nn.Upsample(size=(pull_size, pull_size),
                                      mode='bilinear',
                                      align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # with torch.no_grad():
        if self.scale_factor == 1:
            pull_weights = self.push.weight
        else:
            pull_weights = self.up_sampler(self.push.weight)
        # pull_weights.requires_grad = False

        bias = self.push.bias
        if self.push.bias is not None:
            bias = -self.push.bias

        push = self.relu(self.push(x))
        pull = self.relu(F.conv2d(x,
                                  -pull_weights,
                                  bias,
                                  self.push.stride,
                                  self.pull_padding, self.push.dilation,
                                  self.push.groups))

        alpha = self.alpha
        if self.train_alpha:
            # alpha is greater or equal than 0
            alpha = self.relu(self.alpha)

        if self.dual_output:
            x = torch.cat([push, pull], dim=1)
        else:
            x = push - alpha * pull
            # + self.bias.reshape(1, self.push.out_channels, 1, 1) #.repeat(s[0], 1, s[2], s[3])
        return x

