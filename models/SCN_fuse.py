import torch
import torch.nn as nn
from models.Blocks import *
from models.UNet import *


import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass, binary_erosion, zoom, gaussian_filter

def to_numpy(x):
    x = x.cpu()
    x = x.detach().numpy().astype(np.float32)
    return x

def npimage_to_sitk(image: np.array, transpose_axis=True):
    if transpose_axis:
        image = np.transpose(image, (2, 1, 0))
    return sitk.GetImageFromArray(image)

def sitk_to_npimage(image: sitk.Image, transpose_axis=True):
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    if transpose_axis:
        image_array = np.transpose(image_array, (2, 1, 0))
    return image_array


class HLA(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=64) -> None:
        super().__init__()

        self.conv_list = nn.Sequential(
            *[self.conv_stream(in_channels, f_maps, 2), self.conv_downsample(f_maps, 1)]
        )

        self.hla_backbone1 = UNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps,
                                    layer_order="cbl",
                                    final_activation="tanh",
                                    num_levels=5,
                                    use_attn=False)
        
        self.hla_backbone2 = UNet3D(in_channels=f_maps, out_channels=out_channels, f_maps=f_maps,
                                    layer_order="cbl",
                                    final_activation="tanh",
                                    num_levels=4,
                                    use_attn=False)

    def conv_stream(self, in_channels, f_maps, count=2):

        conv3x3x3 = []
        for i in range(count):

            if i == 0:
                in_channels = in_channels
            else:
                in_channels = f_maps

            conv3x3x3.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, f_maps, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(f_maps),
                    nn.LeakyReLU(inplace=True)
                )
            )

        return nn.Sequential(*conv3x3x3)

    def conv_downsample(self, f_maps, count=1):

        conv3x3x3_s = []
        for _ in range(count):
            conv3x3x3_s.append(
                nn.Sequential(
                    nn.Conv3d(f_maps, f_maps, kernel_size=3,
                              stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(f_maps),
                    nn.LeakyReLU(inplace=True)
                )
            )

        return nn.Sequential(*conv3x3x3_s)

    def forward(self, x):

        x1 = x
        x2 = self.conv_list(x)

        x1_out = self.hla_backbone1(x1)
        x2_out = self.hla_backbone2(x2)

        return x1_out, x2_out


class HSC(nn.Module):
    def __init__(self, io_channels, fmaps=64) -> None:
        super().__init__()

        self.stage_1 = self.make_stage(feature_list=[io_channels, fmaps, io_channels])
        self.stage_2 = self.make_stage(feature_list=[io_channels, fmaps, fmaps, io_channels])
        self.stage_3 = self.make_stage(feature_list=[io_channels, fmaps, fmaps*2, fmaps, io_channels])
        self.conv1x1x1 = nn.Conv3d(io_channels*2, io_channels, 1)
        self.act = nn.Tanh()

    def make_stage(self, feature_list):

        block_list = []

        for i in range(len(feature_list) - 1):

            in_c = feature_list[i]
            out_c = feature_list[i+1]

            block_list.append(
                nn.Sequential(
                    nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False),
                    nn.BatchNorm3d(out_c),
                    nn.LeakyReLU(inplace=True)
                )
            )

        return nn.Sequential(*block_list)

    def forward(self, x):
        
        x_stage_1 = x
        
        t_stage1 = to_numpy(x_stage_1)[0]
        t_stage1 = np.sum(t_stage1, axis=0)
        t_stage1 = zoom(t_stage1, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_stage1), "test_out/stage1.nii.gz")
        
        x_stage_2 = torch.cat([x_stage_1, self.stage_1(x_stage_1)], dim=1)
        x_stage_2 = self.conv1x1x1(x_stage_2)
        
        t_stage2 = to_numpy(x_stage_2)[0] 
        t_stage2 = np.sum(t_stage2, axis=0)
        t_stage2 = zoom(t_stage2, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_stage2), "test_out/stage2.nii.gz")
        

        x_stage_3 = torch.cat([x_stage_2, self.stage_2(x_stage_2)], dim=1)
        x_stage_3 = self.conv1x1x1(x_stage_3)
        
        t_stage3 = to_numpy(x_stage_3)[0] 
        t_stage3 = np.sum(t_stage3, axis=0)
        t_stage3 = zoom(t_stage3, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_stage3*-1), "test_out/stage3.nii.gz")

        x_stage_4 = torch.cat([x_stage_3, self.stage_3(x_stage_3)], dim=1)
        x_stage_4 = self.conv1x1x1(x_stage_4)
        
        t_stage4 = to_numpy(x_stage_4)[0] 
        t_stage4 = np.sum(t_stage4, axis=0)
        t_stage4 = zoom(t_stage4, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_stage4*-1), "test_out/stage4.nii.gz")

        x_stage_4 = self.act(x_stage_4)
        
        t_stage4_act = to_numpy(x_stage_4)[0] 
        t_stage4_act = np.sum(t_stage4_act, axis=0)
        t_stage4_act = zoom(t_stage4_act, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_stage4_act*-1), "test_out/stage4_act.nii.gz")
        
        return x_stage_4


class SCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, f_maps: int = 64) -> None:
        super().__init__()

        self.hla_backbone = HLA(in_channels, out_channels, f_maps)
        self.hsc_backbone = HSC(out_channels, f_maps)
        
        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.up2x2x2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):

        hla, hla_down_hat = self.hla_backbone(x)

        hla_down = self.down(hla)
        
        hsc = self.up2x2x2(
            torch.max(
                torch.abs(hla_down_hat), torch.abs(self.hsc_backbone(hla_down))
                )
            )
        
        out = hla * hsc

        return out


if __name__ == "__main__":
    for i in range(1):
        image_file = torch.randn(1, 1, 64, 64, 224).cuda()
        model = SCN(in_channels=1, out_channels=25, f_maps=64).cuda()
        print(model.__class__.__name__)
        print(model)
        out = model(image_file)
        print(out.shape)
