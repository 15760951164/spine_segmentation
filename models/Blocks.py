from functools import partial
import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib
import SimpleITK as sitk
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage
import numpy as np

# debug
def to_numpy(x):
    x = x.cpu()
    x = x.detach().numpy().astype(np.float32)
    return x


class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio=1/4, pooling_type='att', fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, h, w, z = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, h * w * z)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, h * w * z)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):

        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:

            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

class AttentionLayer3D(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionLayer3D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ChannelSELayer3D(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(
            self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(
            batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):

        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
    


class CRU(nn.Module):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1/2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha*op_channel)
        self.low_channel = low_channel = op_channel-up_channel
        self.squeeze1 = nn.Conv3d(
            up_channel, up_channel//squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(
            low_channel, low_channel//squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv3d(up_channel//squeeze_radio, op_channel, kernel_size=group_kernel_size,
                             stride=1, padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv3d(up_channel//squeeze_radio,
                              op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv3d(low_channel//squeeze_radio, op_channel -
                              low_channel//squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1)//2, dim=1)
        return out1+out2


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight/sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(
            reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(
            reweigts), reweigts)
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([x_11+x_22, x_12+x_21], dim=1)


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1/2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class SingleConv(nn.Sequential):
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        dropout_prob (float): dropout probability, default 0.1
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbl', num_groups=8,
                 padding=1, dropout_prob=0.1, is3d=True):
        super(SingleConv, self).__init__()

        for name, module in self.create_conv(in_channels, out_channels, kernel_size, order,
                                             num_groups, padding, dropout_prob, is3d):
            self.add_module(name, module)

    def create_conv(self, in_channels, out_channels, kernel_size, order, num_groups, padding,
                    dropout_prob, is3d):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size(int or tuple): size of the convolving kernel
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'gcr' -> groupnorm + conv + ReLU
                'cl' -> conv + LeakyReLU
                'ce' -> conv + ELU
                'bcr' -> batchnorm + conv + ReLU
                'cbrd' -> conv + batchnorm + ReLU + dropout
                'cbrD' -> conv + batchnorm + ReLU + dropout2d
            num_groups (int): number of groups for the GroupNorm
            padding (int or tuple): add zero-padding added to all three sides of the input
            dropout_prob (float): dropout probability
            is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
        """
        assert 'c' in order, "Conv layer MUST be present"
        assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

        modules = []
        for i, char in enumerate(order):
            if char == 'r':
                modules.append(('ReLU', nn.ReLU(inplace=True)))
            elif char == 'l':
                modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
            elif char == 'e':
                modules.append(('ELU', nn.ELU(inplace=True)))
            elif char == 'c':
                # add learnable bias only in the absence of batchnorm/groupnorm
                bias = not ('g' in order or 'b' in order)
                if is3d:
                    conv = nn.Conv3d(in_channels, out_channels,
                                     kernel_size, padding=padding, bias=bias)
                else:
                    conv = nn.Conv2d(in_channels, out_channels,
                                     kernel_size, padding=padding, bias=bias)

                modules.append(('conv', conv))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    num_channels = in_channels
                else:
                    num_channels = out_channels

                # use only one group if the given number of groups is greater than the number of channels
                if num_channels < num_groups:
                    num_groups = 1

                assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
                modules.append(('groupnorm', nn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels)))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is3d:
                    bn = nn.BatchNorm3d
                else:
                    bn = nn.BatchNorm2d

                if is_before_conv:
                    modules.append(('batchnorm', bn(in_channels)))
                else:
                    modules.append(('batchnorm', bn(out_channels)))
            elif char == 'd':
                modules.append(('dropout', nn.Dropout3d(p=dropout_prob)))
            elif char == 'D':
                modules.append(('dropout2d', nn.Dropout2d(p=dropout_prob)))
            else:
                raise ValueError(
                    f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']")

        return modules

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cbl',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # check if dropout_prob is a tuple and if so
        # split it for different dropout probabilities for each convolution.
        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        self.conv1_in_channels, self.conv1_out_channels = conv1_in_channels, conv1_out_channels
        self.conv2_in_channels, self.conv2_out_channels = conv2_in_channels, conv2_out_channels

        self.dropout_prob1 = dropout_prob1
        self.dropout_prob2 = dropout_prob2

        if encoder:

            encoder_conv_list = []
            for i in range(1, 1+repeats):
                encoder_conv_list.append(SingleConv(self.conv1_in_channels, self.conv1_out_channels, kernel_size, order, num_groups,
                                                    padding=padding, dropout_prob=self.dropout_prob1, is3d=is3d))

            self.conv1 = nn.Sequential(*encoder_conv_list)

            self.conv2 = nn.Sequential(SingleConv(self.conv2_in_channels, self.conv2_out_channels, kernel_size, order, num_groups,
                                       padding=padding, dropout_prob=self.dropout_prob2, is3d=is3d))

        else:

            self.conv1 = nn.Sequential(SingleConv(self.conv1_in_channels, self.conv1_out_channels, kernel_size, order, num_groups,
                                       padding=padding, dropout_prob=self.dropout_prob1, is3d=is3d))
            deconder_conv_list = []
            for i in range(1, 1+repeats):
                deconder_conv_list.append(SingleConv(self.conv2_in_channels, self.conv2_out_channels, kernel_size, order, num_groups,
                                                     padding=padding, dropout_prob=self.dropout_prob2, is3d=is3d))

            self.conv2 = nn.Sequential(*deconder_conv_list)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class DoubleConvSE(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, se_module="scse", kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(
                num_channels=self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.se_module(x)

        return x

class DoubleConvSC(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        self.sc_model = ScConv(self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.sc_model(x)

        return x


class DoubleConvResidual(DoubleConv):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)
        
            
        if self.conv1_in_channels == self.conv2_out_channels:
            self.residual_edge = nn.Identity()
        else:
            self.residual_edge = nn.Sequential(nn.Conv3d(
                    self.conv1_in_channels,
                    self.conv2_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),nn.BatchNorm3d(self.conv2_out_channels))

        if 'l' in order:
            self.non_linearity=nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
        elif 'e' in order:
            self.non_linearity=nn.ELU(inplace=True)

        else:
            self.non_linearity=nn.ReLU(inplace=True)
            
        self.residual_edge.add_module(name="non_linearity", module=self.non_linearity)

        if 'd' in order:
            self.residual_edge.add_module(name="dropout", module=nn.Dropout3d(p=dropout_prob))
                

    def forward(self, x):

        residual = self.residual_edge(x)

        out = self.conv1(x)
        out = self.conv2(out)

        return self.non_linearity(out + residual)

class DoubleConvResidualSE(DoubleConvResidual):

    def __init__(self, in_channels, out_channels, encoder, se_module="scse", kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=self.conv2_out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(
                num_channels=self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.se_module(x)

        return x


class DoubleConvResidualSC(DoubleConvResidual):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        self.sc_model = ScConv(self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.sc_model(x)

        return x
    
class DoubleConvGC(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        self.gc_model = ContextBlock(self.conv2_out_channels)

    def forward(self, x):

        x = super().forward(x)
        x = self.gc_model(x)
    
        return x
    
class DoubleConvCB(DoubleConv):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1, upscale=2, dropout_prob=0.1, repeats=1, is3d=True):
        super().__init__(in_channels, out_channels, encoder, kernel_size,
                         order, num_groups, padding, upscale, dropout_prob, repeats, is3d)

        if self.conv1_in_channels == self.conv2_out_channels:
            self.residual_edge = nn.Identity()
        else:
            self.residual_edge = nn.Conv3d(self.conv1_in_channels, self.conv2_out_channels,1)

        if 'l' in order:
            self.non_linearity=nn.LeakyReLU(negative_slope=0.1, inplace=True)
            
        elif 'e' in order:
            self.non_linearity=nn.ELU(inplace=True)

        else:
            self.non_linearity=nn.ReLU(inplace=True)
            
        self.cb_model_1 = ContextBlock(self.conv1_out_channels)
        self.cb_model_2 = ContextBlock(self.conv2_out_channels)

    def forward(self, x):

        residual = self.residual_edge(x)

        out = self.conv1(x)
        out = self.cb_model_1(out)
        
        out = self.conv2(out)
        out = self.cb_model_2(out)
        
        out = self.non_linearity(out + residual)

        return out

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, is3d=True, repeats=1, **kwargs):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                is3d=is3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, is3d=is3d)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, se_module='scse', repeats=1, **kwargs):
        super(ResNetBlockSE, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse', 'cse', 'sse']
        if se_module == 'scse':
            self.se_module = ChannelSpatialSELayer3D(
                num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'cse':
            self.se_module = ChannelSELayer3D(
                num_channels=out_channels, reduction_ratio=1)
        elif se_module == 'sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, repeats=1,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.1, is3d=True):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         repeats=repeats,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         upscale=upscale,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2, basic_module=DoubleConv, repeats=1,
                 conv_layer_order='gcr', num_groups=8, padding=1, upsample='default', use_attn=False,
                 dropout_prob=0.1, is3d=True):
        super(Decoder, self).__init__()

        # perform concat joining per default
        concat = True

        # don't adapt channels after join operation
        adapt_channels = False

        self.attention = None

        if upsample is not None and upsample != 'none':
            if upsample == 'default':

                if basic_module == ResNetBlock or basic_module == ResNetBlockSE:
                    upsample = 'deconv'  # use deconvolution upsampling
                    concat = False  # use summation joining
                    adapt_channels = True  # adapt channels after joining
                else:
                    upsample = 'nearest'  # use nearest neighbor interpolation for upsampling
                    concat = True  # use concat joining
                    adapt_channels = False  # don't adapt channels

            # perform deconvolution upsampling if mode is deconv
            if upsample == 'deconv':
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor,
                                                          is3d=is3d)
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        # perform joining operation
        self.joining = partial(self._joining, concat=concat)

        # adapt the number of in_channels for the ResNetBlock
        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         repeats=repeats,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         dropout_prob=dropout_prob,
                                         is3d=is3d)

        if use_attn:
            if not adapt_channels:
                self.attention = AttentionLayer3D(
                    out_channels, in_channels - out_channels, out_channels)
            else:
                self.attention = AttentionLayer3D(
                    in_channels, in_channels, in_channels)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)

        if self.attention is not None:
            encoder_features = self.attention(g=x, x=encoder_features)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):

    class Upsample(nn.Module):

        def __init__(self, conv_transposed, is3d):
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            x = self.conv_transposed(x)
            if self.is3d:
                output_size = x.size()[-3:]
            else:
                output_size = x.size()[-2:]
            if output_size != size:
                return F.interpolate(x, size=size)
            return x

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        if is3d is True:
            conv_transposed = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        else:
            conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=scale_factor, padding=1, bias=False)
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x