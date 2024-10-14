
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.UNet import *
import SimpleITK as sitk
from skimage import filters
import networkx as nx
import matplotlib.pyplot as plt
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage
import numpy as np
from PIL import Image
import sys
from math import sqrt
from transformations.intensity.np.normalize import *


model=UNet3D_ResidualSE(in_channels=1,
                        out_channels=1,
                        f_maps=32,
                        layer_order="cbr",
                        repeats=1,
                        final_activation="sigmoid",
                        conv_kernel_size=3,
                        conv_padding=1,
                        use_attn=False,
                        num_levels=5)

state_dict = torch.load(r"result\model_weight\spine\UNet3D_ResidualSE_00025.pth")["state_dict"]
model.load_state_dict(state_dict)

class SemanticSegmentationTarget:
    def __init__(self):
        pass
        
    def __call__(self, model_output):
        
        new_out = model_output.clone()
        # new_out[new_out > 0.5] = 1
        # new_out[new_out <= 0.5] = 0
        
        return new_out.sum()

image = (sitk_to_npimage(sitk.ReadImage(r"GL003\img\GL003_0015.nii.gz")))
image = torch.from_numpy(image)
image = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])


out = model(image)
out = out.detach().numpy()[0, 0]

target_layers = [model.decoders[3]]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=image, targets=[SemanticSegmentationTarget()])

grayscale_cam = np.max(grayscale_cam[0], axis=1)
image = np.max(image.detach().numpy()[0, 0], axis=1)

image = np.rot90(image)
grayscale_cam = np.rot90(grayscale_cam)

image = np.stack([image]*3, axis=-1)
image = normalize(image, (0,1))

cam_image = show_cam_on_image(img=image, mask=grayscale_cam, use_rgb=True)
Image.fromarray(cam_image).save("test_out/cam_image.jpg")

# sitk.WriteImage(npimage_to_sitk(grayscale_cam[0]), "test_out/grayscale_cam.nii.gz")
# sitk.WriteImage(npimage_to_sitk(out), "test_out/out__1.nii.gz")

pass