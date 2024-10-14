from functools import partial
import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib
from models.Blocks import *

class AbstractUNet(nn.Module):

    def __init__(self, in_channels, out_channels, basic_module, final_activation='sigmoid', f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_type='avg', pool_kernel_size=2, repeats=1, use_attn=False,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            self.f_maps = self.number_of_features_per_level(
                f_maps, num_levels=num_levels)

        assert isinstance(self.f_maps, list) or isinstance(self.f_maps, tuple)
        assert len(self.f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders, self.encoders_maps = self.create_encoders(in_channels, self.f_maps, basic_module, conv_kernel_size,
                                             conv_padding, conv_upscale, dropout_prob,
                                             layer_order, num_groups, pool_kernel_size, is3d, repeats,pool_type)

        # create decoder path
        self.decoders, self.decoders_maps = self.create_decoders(self.f_maps, basic_module, conv_kernel_size, conv_padding,
                                             layer_order, num_groups, upsample, dropout_prob,
                                             is3d, repeats, use_attn)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(self.f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(self.f_maps[0], out_channels, 1)

        self.final_activation = self.activate(final_activation)

    def activate(self, activation):

        if activation is not None:
            if activation == 'relu':
                return nn.ReLU(inplace=True)
            elif activation == 'leakyrelu':
                return nn.LeakyReLU(inplace=True)
            elif activation == 'prelu':
                return nn.PReLU()
            elif activation == 'celu':
                return nn.CELU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()
            elif activation == 'softmax':
                return nn.Softmax(dim=1)
            elif activation == 'tanh':
                return nn.Tanh()
            elif activation == 'softsign':
                return nn.Softsign()
            elif activation == 'hardtanh':
                return nn.Hardtanh(min_val=0.0, max_val=1.0)
            else:
                raise NotImplementedError(
                    'Option {} not implemented. Available options: relu | leakyrelu | prelu | celu | sigmoid | softmax ;'.format(activation))
        else:
            return None

    def number_of_features_per_level(self, init_channel_number, num_levels):
        return [init_channel_number * 2 ** k for k in range(num_levels)]

    def create_encoders(self, in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
                        conv_upscale, dropout_prob,
                        layer_order, num_groups, pool_kernel_size, is3d, repeats, pool_type):
        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        out_feature_list = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                # apply conv_coord only in the first encoder if any
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the firs encoder
                                  basic_module=DoubleConv,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding,
                                  upscale=conv_upscale,
                                  dropout_prob=dropout_prob,
                                  is3d=is3d,
                                  repeats=1,
                                  pool_type=pool_type)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding,
                                  upscale=conv_upscale,
                                  dropout_prob=dropout_prob,
                                  is3d=is3d,
                                  repeats=repeats,
                                  pool_type=pool_type)

            encoders.append(encoder)
            out_feature_list.append(out_feature_num)
            

        return nn.ModuleList(encoders), out_feature_list

    def create_decoders(self, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                        num_groups, upsample, dropout_prob, is3d, repeats, use_attn):
        # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
        decoders = []
        out_feature_list = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if not (basic_module == ResNetBlock or basic_module == ResNetBlockSE) and upsample != 'deconv':
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]

            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              upsample=upsample,
                              dropout_prob=dropout_prob,
                              is3d=is3d,
                              repeats=repeats,
                              use_attn=use_attn,
                              )
            decoders.append(decoder)
            out_feature_list.append(out_feature_num)
        return nn.ModuleList(decoders), out_feature_list

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, pool_type='max', use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     repeats=repeats,
                                     use_attn=use_attn,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     final_activation=final_activation,
                                     is3d=True,
                                     pool_type=pool_type)


class UNet3D_SE(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_SE, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        repeats=repeats,
                                        use_attn=use_attn,
                                        basic_module=DoubleConvSE,
                                        f_maps=f_maps,
                                        layer_order=layer_order,
                                        num_groups=num_groups,
                                        num_levels=num_levels,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        conv_upscale=conv_upscale,
                                        upsample=upsample,
                                        dropout_prob=dropout_prob,
                                        final_activation=final_activation,
                                        is3d=True)
        
        self.final_conv = nn.Sequential(
            Encoder(self.f_maps[0], self.f_maps[0],
                    apply_pooling=False,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    padding=conv_padding,
                    upscale=conv_upscale,
                    dropout_prob=dropout_prob,
                    is3d=True,
                    repeats=1,
                    pool_type="max"),
            nn.Conv3d(self.f_maps[0], out_channels, 1)
        )

        self.final_activation = self.activate(final_activation)


class UNet3D_SC(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_SC, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        repeats=repeats,
                                        use_attn=use_attn,
                                        basic_module=DoubleConvSC,
                                        f_maps=f_maps,
                                        layer_order=layer_order,
                                        num_groups=num_groups,
                                        num_levels=num_levels,
                                        conv_kernel_size=conv_kernel_size,
                                        conv_padding=conv_padding,
                                        conv_upscale=conv_upscale,
                                        upsample=upsample,
                                        dropout_prob=dropout_prob,
                                        final_activation=final_activation,
                                        is3d=True)


class UNet3D_CB(AbstractUNet):
    def __init__(self, in_channels, out_channels, basic_module=DoubleConvCB, final_activation='sigmoid', f_maps=64, layer_order='gcr', num_groups=8, num_levels=4, conv_kernel_size=3, pool_type='max', pool_kernel_size=2, repeats=1, use_attn=False, conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super().__init__(in_channels, out_channels, basic_module, final_activation, f_maps, layer_order, num_groups, num_levels, conv_kernel_size, pool_type, pool_kernel_size, repeats, use_attn, conv_padding, conv_upscale, upsample, dropout_prob, is3d)
        
        
        self.final_conv = nn.Sequential(
            Encoder(self.f_maps[0], self.f_maps[0],
                    apply_pooling=False,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    padding=conv_padding,
                    upscale=conv_upscale,
                    dropout_prob=dropout_prob,
                    is3d=is3d,
                    repeats=1,
                    pool_type=pool_type),
            nn.Conv3d(self.f_maps[0], out_channels, 1)
        )

        self.final_activation = self.activate(final_activation)

class UNet3D_Residual(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_Residual, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              repeats=repeats,
                                              use_attn=use_attn,
                                              basic_module=DoubleConvResidual,
                                              f_maps=f_maps,
                                              layer_order=layer_order,
                                              num_groups=num_groups,
                                              num_levels=num_levels,
                                              conv_kernel_size=conv_kernel_size,
                                              conv_padding=conv_padding,
                                              conv_upscale=conv_upscale,
                                              upsample=upsample,
                                              dropout_prob=dropout_prob,
                                              final_activation=final_activation,
                                              is3d=True)


class UNet3D_ResidualSE(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_ResidualSE, self).__init__(in_channels=in_channels,
                                                out_channels=out_channels,
                                                repeats=repeats,
                                                use_attn=use_attn,
                                                basic_module=DoubleConvResidualSE,
                                                f_maps=f_maps,
                                                layer_order=layer_order,
                                                num_groups=num_groups,
                                                num_levels=num_levels,
                                                conv_kernel_size=conv_kernel_size,
                                                conv_padding=conv_padding,
                                                conv_upscale=conv_upscale,
                                                upsample=upsample,
                                                dropout_prob=dropout_prob,
                                                final_activation=final_activation,
                                                is3d=True)


class UNet3D_ResidualSC(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', repeats=1, use_attn=False,
                 num_groups=8, num_levels=5, conv_kernel_size=3, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D_ResidualSC, self).__init__(in_channels=in_channels,
                                                out_channels=out_channels,
                                                repeats=repeats,
                                                use_attn=use_attn,
                                                basic_module=DoubleConvResidualSC,
                                                f_maps=f_maps,
                                                layer_order=layer_order,
                                                num_groups=num_groups,
                                                num_levels=num_levels,
                                                conv_kernel_size=conv_kernel_size,
                                                conv_padding=conv_padding,
                                                conv_upscale=conv_upscale,
                                                upsample=upsample,
                                                dropout_prob=dropout_prob,
                                                final_activation=final_activation,
                                                is3d=True)


class ResidualUNet3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', use_attn=False,
                 num_groups=8, num_levels=5, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             use_attn=use_attn,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             final_activation=final_activation,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):

    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr', use_attn=False,
                 num_groups=8, num_levels=5, conv_padding=1, final_activation="sigmoid",
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               use_attn=use_attn,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               final_activation=final_activation,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True)


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    for i in range(10):
        torch.cuda.empty_cache()
        image = torch.randn(1, 1, 96, 96, 96).cuda()
        model = UNet3D_Residual(in_channels=1, out_channels=1, f_maps=32, layer_order="cbrd", repeats=1,pool_type='max',
                final_activation="sigmoid", conv_kernel_size=3, conv_padding=1, use_attn=False, num_levels=5).cuda()
        
        print(model.__class__.__name__)
        #print(model)
        out = model(image)
        print(out.shape)
