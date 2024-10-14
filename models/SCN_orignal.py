import torch
import torch.nn as nn
from models.UNet import *
from models.Blocks import DoubleConv, SingleConv
#from UNet import *

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class HLA_EncoderModel(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.conv1 = DoubleConv(in_channels, out_channels, False)
        self.conv2 = SingleConv(out_channels, out_channels)
        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)
    
    def forward(self, x):
        
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        
        x_1_down = self.down(x_1)
        
        return x_1_down, x_2
    
class HLA_DecoderModel(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.conv = SingleConv(in_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    
    def forward(self, x_down, x):
        
        x_up = self.up(x_down)
        x_fuse = torch.cat([x, x_up], dim=1)
        out = self.conv(x_fuse)
        
        return out

class HLA(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=16, num_levels=5) -> None:
        super().__init__()
        
        self.f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        self.encoder0 = DoubleConv(in_channels=in_channels, out_channels=self.f_maps[0], encoder=False, order="cbl")
        self.encoder1 = HLA_EncoderModel(self.f_maps[0], self.f_maps[1])
        self.encoder2 = HLA_EncoderModel(self.f_maps[1], self.f_maps[2])
        self.encoder3 = HLA_EncoderModel(self.f_maps[2], self.f_maps[3])
        self.encoder4 = HLA_EncoderModel(self.f_maps[3], self.f_maps[4])
        
        self.decoder3 = HLA_DecoderModel(self.f_maps[3] + self.f_maps[4], self.f_maps[3])
        self.decoder2 = HLA_DecoderModel(self.f_maps[2] + self.f_maps[3], self.f_maps[2])
        self.decoder1 = HLA_DecoderModel(self.f_maps[1] + self.f_maps[2], self.f_maps[1])
        self.decoder0 = DoubleConv(in_channels=self.f_maps[1], out_channels=self.f_maps[0], encoder=False, order="cbl")

        self.final_conv = nn.Sequential(nn.Conv3d(self.f_maps[0], out_channels, 1), nn.Tanh())
        
    def forward(self, x):
        
        x0 = self.encoder0(x)
        x1_down, x1 = self.encoder1(x0)
        x2_down, x2 = self.encoder2(x1_down)
        x3_down, x3 = self.encoder3(x2_down)
        _, x4 = self.encoder4(x3_down)
        
        x3_de = self.decoder3(x4, x3)
        x2_de = self.decoder2(x3_de, x2)
        x1_de = self.decoder1(x2_de, x1)
        
        out = self.final_conv(self.decoder0(x1_de))
        
        return out

class HSC(nn.Module):
    def __init__(self, io_channels, fmaps=64, num_levels=4) -> None:
        super().__init__()
        
        self.f_maps = number_of_features_per_level(fmaps, num_levels=num_levels)
        self.f_maps.insert(0, io_channels)
        
        self.stage = self.make_stage(self.f_maps)
        
        self.final_conv = nn.Conv3d(self.f_maps[-1], io_channels, 1)
        
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
        
        x = self.stage(x)
        x = self.final_conv(x)
        
        return x
            
class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        f_maps: int = 32,
    ):
        super().__init__()

        self.scnet_local = HLA(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps)

        self.local_heatmaps = nn.Identity()

        self.down = nn.MaxPool3d(2, 2, ceil_mode=True)

        self.scnet_spatial = HSC(io_channels=out_channels)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Tanh()
        )

        self.spatial_heatmaps = nn.Identity()

    def forward(self, inputs):

        node = self.scnet_local(inputs)

        local_heatmaps = node = self.local_heatmaps(node)

        node = self.down(node)

        node = self.scnet_spatial(node)

        node = self.spatial_heatmaps(node)

        spatial_heatmaps = self.up(node)

        heatmaps = local_heatmaps * spatial_heatmaps

        return heatmaps

if __name__ == "__main__":
    for i in range(10):
        image_file = torch.randn(1, 1, 96, 96, 96).cuda()
        model = SCN(in_channels=1, out_channels=25, f_maps=32).cuda()
        print(model.__class__.__name__)
        # print(model)
        out = model(image_file)
        print(out.shape)
