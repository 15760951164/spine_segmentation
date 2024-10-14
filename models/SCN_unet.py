import torch
import torch.nn as nn
from typing import Sequence
from models.UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D
# from UNet import UNet3D_Residual, UNet3D_ResidualSE, UNet3D_SE, UNet3D


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


class SCN_UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_maps: int = 32,
        num_levels: int = 5
    ):
        super().__init__()

        self.scnet_local = UNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps,
                                  num_levels=num_levels, layer_order="cbl", pool_type='max', repeats=1, final_activation="tanh", use_attn=False)

        self.local_heatmaps = nn.Identity()

        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)

        self.scnet_spatial = UNet3D(in_channels=out_channels, out_channels=out_channels, f_maps=f_maps,
                                    num_levels=num_levels, layer_order="cbl", pool_type='max', repeats=0, final_activation=None, use_attn=False)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Tanh()
        )

        self.spatial_heatmaps = nn.Identity()

    def forward(self, inputs):
        
        t_image = to_numpy(inputs)[0][0]
        t_image = zoom(t_image, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_image), "test_out/image.nii.gz")

        node = self.scnet_local(inputs)

        local_heatmaps = node = self.local_heatmaps(node)

        t_local_heatmaps = to_numpy(local_heatmaps)[0] 
        t_local_heatmaps = np.sum(t_local_heatmaps, axis=0)
        t_local_heatmaps = zoom(t_local_heatmaps, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(np.abs(t_local_heatmaps)), "test_out/local_heatmaps.nii.gz")

        node = self.down(node)

        node = self.scnet_spatial(node)

        node = self.spatial_heatmaps(node)

        spatial_heatmaps = self.up(node)
        
        t_spatial_heatmaps = to_numpy(spatial_heatmaps)[0]
        t_spatial_heatmaps = np.sum(t_spatial_heatmaps, axis=0)
        t_spatial_heatmaps = zoom(t_spatial_heatmaps, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(np.abs(t_spatial_heatmaps)), "test_out/spatial_heatmaps.nii.gz")

        heatmaps = local_heatmaps * spatial_heatmaps
        
        t_heatmaps = to_numpy(heatmaps)[0]
        t_heatmaps = np.sum(t_heatmaps, axis=0)
        t_heatmaps = zoom(t_heatmaps, zoom=(2, 2, 2), order=1)
        sitk.WriteImage(npimage_to_sitk(t_heatmaps), "test_out/heatmaps.nii.gz")

        return heatmaps


if __name__ == "__main__":
    image = torch.randn(1, 1, 96, 96, 128)
    model = SCN_UNet(in_channels=1, out_channels=25, f_maps=32)
    print(model.__class__.__name__)
    print(model)
    out = model(image)
    print(out.shape)
