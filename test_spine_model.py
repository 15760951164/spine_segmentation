import numpy as np
from inference import Landmark
import json
from models.UNet import *
from models import SCN_fuse
from test import write_coords_to_json
import os
from glob import glob
from inference import *

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

model_file = r"result\model_weight\test_model\spine_unetcb\min_loss_weight.pth"
model = UNet3D_CB(in_channels=1, out_channels=1, f_maps=32, layer_order="cbl", final_activation="sigmoid", num_levels=5, use_attn=False)
image_folder = "test_data/Verse2020"
model_folder = "result/model_weight/test_model/scn"
model_files = glob(os.path.join(model_folder, "SCN*.pth"))

for model_file in model_files:

    spine_seg = SpineSegmention(model_file=model_file,
                                model_func=model
                                )


    output_folder = "spine_seg/spine_seg_test"

    postproc_image_list = []
    check_dir(output_folder)
    for id in os.listdir(image_folder):
        filenames = glob(os.path.join(image_folder, id, '*.nii.gz'))
        for filename in sorted(filenames):

            output_path = os.path.join(output_folder, id)
            check_dir(output_path)
            
            for filename in sorted(filenames):
                if "_seg" not in filename:
                    image_file = filename

        spine_seg.inference(inference_image=image_file, save_path=output_path)

        pass