
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
from utils.sitk_image import resample_to_spacing
from utils.heatmap import get_local_maxima


a = np.linalg.norm(np.array([39,42,28])-np.array([39,46,25]))
b = np.linalg.norm(np.array([39,42,28])-np.array([39,52,18]))

heatmap = sitk_to_npimage(sitk.ReadImage(r"landmark\landmark_test\GL216\heatmap_25_channel.nii.gz"))
heatmap = np.clip(heatmap, 0.3, 1)
print(get_local_maxima(heatmap)[0])