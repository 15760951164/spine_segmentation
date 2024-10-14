import numpy as np
from inference import Landmark
import json
from models.UNet import *
import os
from inference import VertebraSegmention
from glob import glob
from utils.sitk_image import resample_to_spacing
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from transformations.intensity.np.normalize import *
from PIL import Image
import re
import multiprocessing

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_coords(json_file):
    with open(json_file, 'r') as f:
        data = f.read()
        anno = json.loads(data)
        locs = []
        for i in range(len(anno)):
            label = int(anno[i]['label'])
            x = int(anno[i]['X'])
            y = int(anno[i]['Y'])
            z = int(anno[i]['Z'])
            locs.append(
                Landmark(coords=np.array([x, y, z]),
                        is_valid=True,
                        label=label,
                        scale=1,
                        value=0))
    return locs

def test(vert_seg, output_folder, image_folder):
    
    check_dir(output_folder)

    for id in os.listdir(image_folder):
        
        filenames = glob(os.path.join(image_folder, id, '*'))
        
        for filename in sorted(filenames):
            if "1mm" in filename:
                json_file = filename
            elif "_seg" not in filename and "nii.gz" in filename and "heatmap" not in filename:
                image_file = filename
        
        coords = read_coords(json_file)
        output_path = os.path.join(output_folder, id)
        check_dir(output_path)
        
        vert_seg.inference(inference_image=image_file, save_path=output_path, coord_info=coords, filter=False)

# model = UNet3D_CB(in_channels=2, out_channels=1, f_maps=64, layer_order="cbl", final_activation="sigmoid", num_levels=5, use_attn=False)
# model = UNet3D(in_channels=1, out_channels=1, f_maps=16, layer_order="cbl", repeats=1, final_activation="sigmoid", use_attn=False, num_levels=5)
# model = UNet3D(in_channels=2, out_channels=1, f_maps=16, layer_order="cbl", repeats=1, final_activation="sigmoid", use_attn=False, num_levels=5)
# model_file = r"result\model_weight\test_model\vertebra_unet\min_loss_weight.pth"

if __name__=="__main__":
    
    model = UNet3D_CB(in_channels=2, out_channels=1, f_maps=16, layer_order="cbl", final_activation="sigmoid", num_levels=5, use_attn=False)
    
    image_folder = "test_data/Verse2020"
    model_folder = r"result\model_weight\test_model\vertebra_unetatten_96_offset_15.0"
    model_files = glob(os.path.join(model_folder, "*UNet*.pth"))

    func_args = []

    for model_file in model_files:

        curr_epoch = [int(s) for s in re.findall(r'\d+', model_file)][-1]
        
        if curr_epoch not in [50]:
            continue
        
        output_folder = f"idv_segment_locate_scn_unetcb_112_offset/test_epoch_{curr_epoch:05d}"
        
        vert_seg = VertebraSegmention(
            model_func=model,
            sigmas=15.0,
            model_file=model_file)
        vert_seg.set_cube_size((96, 96, 96))
        test(vert_seg, output_folder, image_folder)
    #   func_args.append((vert_seg, output_folder, image_folder))
        
    # pool = multiprocessing.Pool(4)
    # pool.starmap(test, func_args)