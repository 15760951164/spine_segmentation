import numpy as np
from inference import Landmark
import json
from models.UNet import *
from models import SCN_fuse, SCN_orignal, SCN_unet
from test import write_coords_to_json
import os
from glob import glob
from inference import *
import re
import multiprocessing

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_roi_box(binary_mask):

    binary_mask_copy = np.copy(binary_mask)
    binary_mask_copy = binary_erosion(binary_mask_copy).astype(
        binary_mask_copy.dtype)
    x_array, y_array, z_array = np.where(binary_mask_copy > 0)

    x_min, x_max = np.min(x_array), np.max(x_array)
    y_min, y_max = np.min(y_array), np.max(y_array)
    z_min, z_max = np.min(z_array), np.max(z_array)

    roi_box = Box(min_coords=np.array([x_min, y_min, z_min]),
                    max_coords=np.array([x_max, y_max, z_max]))

    return roi_box

def test(loacte_muti, output_folder, image_folder):
    
    check_dir(output_folder)
        
    for id in os.listdir(image_folder):
        if id != "verse603":
            continue
        filenames = glob(os.path.join(image_folder, id, '*.nii.gz'))
        
        for filename in sorted(filenames):

            output_path = os.path.join(output_folder, id)
            check_dir(output_path)
        
            for filename in sorted(filenames):
                if "_seg" in filename:
                    seg_file = filename
                elif "heatmap" not in filename:
                    image_file = filename
            
        roi_box = get_roi_box(sitk_to_npimage(resample_to_spacing(sitk.ReadImage(seg_file))))
        
        heatmap, coords_info_1mm, coords_info_2mm = loacte_muti.inference(inference_image=image_file, save_path=output_path, roi_box=roi_box)
        
        write_coords_to_json(os.path.join(output_path, "landmark_1mm.json"), coords_info_1mm)
        write_coords_to_json(os.path.join(output_path, "landmark_2mm.json"), coords_info_2mm)
        
        
# model = SCN_fuse.SCN(in_channels=1, out_channels=25, f_maps=64)
# model = SCN_orignal.SCN(in_channels=1, out_channels=25, f_maps=32)
# model = UNet3D(in_channels=1, out_channels=25, f_maps=32, layer_order="cbl", final_activation="tanh", num_levels=5, use_attn=False)

if __name__=="__main__":

    #model = SCN_unet.SCN_UNet(in_channels=1, out_channels=25, f_maps=32)
    model = SCN_fuse.SCN(in_channels=1, out_channels=25, f_maps=32)
    
    image_folder = "test_data/Verse2020"
    model_folder = "hyy/scn_fuse_hyy"

    model_files = glob(os.path.join(model_folder, "*SCN*.pth"))

    func_args = []

    for model_file in model_files:

        curr_epoch = [int(s) for s in re.findall(r'\d+', model_file)][-1]
        
        if curr_epoch not in [160]:
            continue

        loacte_muti = VertebraLocate_25_Channel(model_file=model_file,
                                                model_func=model,
                                                min_landmark_value=0.15, 
                                                #postprocess_func=None
                                                )
        
        output_folder = f"hyy/scn_fuse_hyy/test_out/graph/epoch_{curr_epoch:05d}"
        
        test(loacte_muti, output_folder, image_folder)
        
    #    func_args.append((loacte_muti, output_folder, image_folder))
    # pool = multiprocessing.Pool(4)
    # pool.starmap(test, func_args)