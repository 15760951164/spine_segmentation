import os
import numpy as np
import glob
import json
from inference import *
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib
from utils.landmark.landmark_statistics import LandmarkStatistics
from utils.segmentation.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric, PrecisionMetric, RecallMetric
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from collections import OrderedDict
from utils.io.text import *
from utils.io.landmark import *
import pandas as pd
import multiprocessing

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_annotations_from_json_file(file, scale=1.0):

    with open(file, 'r') as f:
        data = f.read()
        anno = json.loads(data)

    locs = [Landmark(coords=np.array([np.nan, np.nan, np.nan]), is_valid=False)] * 25


    for i in range(len(anno)):
        label = int(anno[i]['label'])
        x = float(anno[i]['X'])
        y = float(anno[i]['Y'])
        z = float(anno[i]['Z'])

        if label >= 25:
            continue
        locs[label-1] = Landmark(coords=np.array([x, y, z]) * scale,
                             is_valid=True, label=label, scale=1, value=0)

    return locs


def get_label(coords_info):
    labels = []
    for i in coords_info:
        labels.append(i.label)
    return labels

def eval_landmark(gt_base_folder, pred_base_folder, output_folder, output_name=None):

    check_dir(output_folder)
    gt = os.listdir(gt_base_folder)
    pred = os.listdir(pred_base_folder)
    landmark_statistics = LandmarkStatistics()

    for index, (gt_folder, pred_folder) in enumerate(zip(gt, pred)):

        assert gt_folder == pred_folder
        
        image_id = gt_folder
        
        gt_folder = os.path.join(gt_base_folder, gt_folder)
        pred_folder = os.path.join(pred_base_folder, pred_folder)
        
        pred_label_path = glob.glob(os.path.join(pred_folder, "*2mm.json"))[0]
        gt_label_path = glob.glob(os.path.join(gt_folder, "*.json"))[0]

        pred_label_info = read_annotations_from_json_file(pred_label_path, scale=1.0)
        gt_label_info = read_annotations_from_json_file(gt_label_path, scale=0.5)

        landmark_statistics.add_landmarks(image_id, pred_label_info, gt_label_info)
    # save_points_csv(landmarks, 'points.csv')
    overview_string = landmark_statistics.get_overview_string([2, 4, 6, 10, 20], 10, 10)
    print(overview_string)
    if output_name == None:
        output_name = 'landmark_eval.txt'
    save_string_txt(overview_string, os.path.join(output_folder, output_name))

def eval_segment(gt_base_folder, pred_base_folder, output_folder):

    check_dir(output_folder)
    
    gt = os.listdir(gt_base_folder)
    pred = os.listdir(pred_base_folder)

    segmentation_statistics = SegmentationStatistics(
        metrics=OrderedDict([('dice', DiceMetric()),
                             (('hau_distance'), HausdorffDistanceMetric()),
                             ('precision', PrecisionMetric()),
                             ('recall', RecallMetric())]))
    
    segmentation_statistics.reset()
    labels = [i for i in range(1, 25)]

    for index, (gt_folder, pred_folder) in enumerate(zip(gt, pred)):
        
        assert gt_folder == pred_folder
    
        print(f"\neval image landmark and segment on {gt_folder}, {index+1}/{len(gt)}...")

        gt_folder = os.path.join(gt_base_folder, gt_folder)
        pred_folder = os.path.join(pred_base_folder, pred_folder)

        for path in glob.glob(os.path.join(pred_folder, "*.nii.gz")):
            if "_seg" in path:
                pred_seg_path = path

        for path in glob.glob(os.path.join(gt_folder, "*.nii.gz")):
            if "_seg" in path:
                gt_seg_path = path

        pred_seg = resample_to_spacing(sitk.ReadImage(pred_seg_path, sitk.sitkInt32), [2.0] * 3)
        gt_seg = resample_to_spacing(sitk.ReadImage(gt_seg_path, sitk.sitkInt32), [2.0] * 3)

        segmentation_statistics.add_labels(gt_folder, pred_seg, gt_seg, labels)
        
    segmentation_statistics.finalize(output_folder)
        
if __name__ == "__main__":
    
    gt_base_folder = "test_data/Verse2020"
    #pred_base_folder = "hyy/scn_unet_hyy_1000/test_out/graph"
    #output_folder = "landmark_scn_fuse/landmark_result"

    p = "hyy/scn_fuse_hyy/test_out/graph"

    for i in os.listdir(p):
        pred_base_folder = os.path.join(p, i)
        output_folder = os.path.join(p)
        output_name = i + "_result.txt"
        eval_landmark(gt_base_folder, pred_base_folder, output_folder, output_name)
    
    # func_args = []
    
    # for i in os.listdir("idv_segment_locate_postproc"):
    #     pred_base_folder = os.path.join("idv_segment_locate_postproc", i)
    #     output_folder = os.path.join("idv_segment_locate_postproc", i + "_result")
    #     func_args.append((gt_base_folder, pred_base_folder, output_folder))
    #     eval_segment(gt_base_folder, pred_base_folder, output_folder)
        
    # pool = multiprocessing.Pool(4)
    # pool.starmap(eval_segment, func_args)