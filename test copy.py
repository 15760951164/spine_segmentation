import os
import json
import copy
from collections import OrderedDict
from glob import glob
import numpy as np
import SimpleITK as sitk
from utils.sitk_np import npimage_to_sitk, sitk_to_npimage
from utils.sitk_image import resample_to_spacing
from scipy.ndimage import center_of_mass, binary_erosion, label
from inference import *
from utils.segmentation.metrics import *
from models import SCN_fuse, SCN_orignal, SCN_unet
from models import encoder3d
import multiprocessing


def read_image(image_path, out_spacing=[1.0, 1.0, 1.0]):
    if image_path is None:
        return

    sitk_image = sitk.ReadImage(image_path)
    resampled_image = resample_to_spacing(sitk_image, out_spacing)
    return sitk_to_npimage(resampled_image)


def read_annotations_from_json_file(file):

    with open(file, "r") as f:
        data = f.read()
        anno = json.loads(data)

    locs = [[] for _ in range(26)]

    for i in range(len(anno)):
        label = int(anno[i]["label"])
        x = int(anno[i]["X"])
        y = int(anno[i]["Y"])
        z = int(anno[i]["Z"])
        locs[label - 1] = Landmark(
            coords=np.array([x, y, z]), is_valid=True, label=label, scale=1, value=0
        )

    for i, loc in enumerate(locs):
        if isinstance(loc, list):
            locs[i] = Landmark(
                coords=np.array([np.nan] * 3), is_valid=False, scale=1, value=0
            )

    return locs


def write_list_to_file(save_path, save_list):
    if len(save_list) == 0:
        return
    with open(save_path, "w") as file:
        for item in save_list:
            file.write(str(item) + "\n")


def find_first(coords_info):
    for index, coord in enumerate(coords_info):
        if coord.is_valid == True:
            return index, coord


def find_last(coords_info):
    for index, coord in enumerate(reversed(coords_info)):
        if coord.is_valid == True:
            return len(coords_info) - index, coord


def pad_first(coords_info, spine_roi, gap_size=20):
    index, coord = find_first(coords_info)
    max_z = spine_roi.max_coords[2]
    pad_count = (max_z - coord.coords[2]) / gap_size
    pad_coords = []
    if pad_count > 1.5:
        pad_count = int(pad_count)
        for i in range(1, pad_count + 1):
            x, y = coord.coords[0], coord.coords[1]
            new_z = coord.coords[2] + gap_size * i
            if new_z > max_z - 10:
                continue
            pad_coords.append(np.array([x, y, new_z]))

    return pad_coords


def pad_last(coords_info, spine_roi, gap_size=20):
    index, coord = find_last(coords_info)
    min_z = spine_roi.min_coords[2]
    pad_count = (coord.coords[2] - min_z) / gap_size
    pad_coords = []
    if pad_count > 1.5:
        pad_count = int(pad_count)
        for i in range(1, pad_count + 1):
            x, y = coord.coords[0], coord.coords[1]
            new_z = coord.coords[2] - gap_size * i
            if new_z < min_z + 10:
                continue
            pad_coords.append(np.array([x, y, new_z]))

    return pad_coords


def pad_middle(coords_info, image_shape, gap_size=20):
    prev_coord = None
    pad_coords = []
    for index, coord in enumerate(coords_info):
        if coord.is_valid == False:
            continue
        current_coord = coord
        cur_x, cur_y, cur_z = current_coord.coords[:]
        if prev_coord is not None:
            prev_x, prev_y, prev_z = prev_coord.coords[:]
            pad_count = (prev_z - cur_z) / gap_size
            if pad_count > 1.6:
                pad_count = int(pad_count)
                x_gap = (cur_x - prev_x) / (pad_count + 1)
                y_gap = (cur_y - prev_y) / (pad_count + 1)
                z_gap = (cur_z - prev_z) / (pad_count + 1)
                for i in range(1, pad_count + 1):
                    new_x, new_y, new_z = (
                        prev_x + x_gap * i,
                        prev_y + y_gap * i,
                        prev_z + z_gap * i,
                    )
                    pad_coords.append(np.array([new_x, new_y, new_z]))
        prev_coord = current_coord

    return pad_coords


def filter_on_diff(diff, threshold=7820, threshold_ratio=0.5):
    coords = []
    diff = binary_erosion(diff).astype(diff.dtype)
    structure = np.ones((3, 3, 3), dtype=np.int32)
    components_labels, ncomponents = label(diff, structure)
    for x in range(1, ncomponents):
        component = components_labels == x

        count = np.count_nonzero(component)
        if count <= threshold * threshold_ratio:
            continue
        coords.append(np.array(list(center_of_mass(component))))
    return coords


def get_coords_array(coords_info):
    ori_coords = []
    for coord in coords_info:
        if coord.is_valid == True:
            ori_coords.append(coord.coords)
    return ori_coords


def updata_coords_ctd(muti_label_mask):

    labels = np.unique(muti_label_mask)[1:]
    new_coords = []
    for index, label in enumerate(labels):
        ctd = center_of_mass((muti_label_mask == label).astype(int))
        new_coords.append(
            Landmark(coords=np.array(list(ctd)), is_valid=True, label=label)
        )

    return new_coords


def update_coords_list(new_coords, coords_info):
    if len(new_coords) == 0:
        return coords_info
    coords_info_copy = copy.deepcopy(coords_info)
    ori_coords = get_coords_array(coords_info)

    for fliter_coord in new_coords:
        norm = np.linalg.norm(ori_coords - fliter_coord, axis=1)
        closest_value = ori_coords[np.argmin(norm)]

        for index, coord in enumerate(coords_info_copy):
            if coord.is_valid == True and np.array_equal(coord.coords, closest_value):

                if fliter_coord[2] > closest_value[2]:
                    coords_info_copy.insert(
                        index, Landmark(coords=fliter_coord, is_valid=True)
                    )
                else:
                    coords_info_copy.insert(
                        index + 1, Landmark(coords=fliter_coord, is_valid=True)
                    )
                ori_coords = get_coords_array(coords_info_copy)
                break

    return coords_info_copy


def print_coords(coords_info):
    for coord in coords_info:
        if coord.is_valid == True:
            print(coord.label, coord.coords)


def manual_resegment_prev(spine_roi, coords_info):

    coords_info = sorted(coords_info, key=lambda x: x.coords[2], reverse=True)
    a = pad_first(coords_info, spine_roi)
    b = pad_last(coords_info, spine_roi)
    # c = pad_middle(coords_info, inference_image.shape)

    return creat_landmark_with_list(a + b)


def creat_landmark_with_list(coords_list):
    coords = []
    for coord in coords_list:
        coords.append(Landmark(coords=np.array(coord), is_valid=True))
    return coords


def make_binary_mask(mask):
    mask_copy = np.copy(mask)
    mask_copy[mask_copy != 0] = 1
    return mask_copy


def auto_resegment_prev(
    muti_segmention: VertebraSegmention,
    image: np.array,
    binary_mask: np.array,
    label_mask: np.array,
):
    binary_mask = make_binary_mask(binary_mask)
    label_mask = make_binary_mask(label_mask)
    new_coords = []

    for _ in range(10):
        diff = binary_mask - label_mask
        diff[diff != 1] = 0
        filter_coords = filter_on_diff(diff, threshold_ratio=0.75)
        if len(filter_coords) == 0:
            break
        filter_coords = creat_landmark_with_list(filter_coords)
        muti_mask, mask_list = muti_segmention.inference(
            inference_image=image, coord_info=filter_coords
        )
        for mask in mask_list:
            new_coords.append(
                Landmark(coords=np.array(list(center_of_mass(mask["mask"]))), is_valid=True)
            )
        label_mask[make_binary_mask(muti_mask) == 1] = 1
        binary_mask = np.logical_or(binary_mask, label_mask).astype(np.int32)

    return new_coords


def get_label(coords_info):
    labels = []
    for i in coords_info:
        labels.append(i.label)
    return labels


def find_max_continuous_coords(coords_list, step=1):

    label_list = get_label(coords_list)
    label_list.append(-999)

    continuous_label = []
    max_continuous_label = []

    for i in range(len(label_list)-1):
        curr_index = i
        next_index = i + 1
        
        if label_list[next_index] == label_list[curr_index] + step:
            continuous_label.append(curr_index)
        else:
            continuous_label.append(curr_index)
            if len(continuous_label) > len(max_continuous_label):
                max_continuous_label = continuous_label.copy()
            continuous_label.clear()

    if len(max_continuous_label) > len(continuous_label):
        return max_continuous_label
    else:
        return continuous_label

def is_same_vertebra(
    muti_segmention: VertebraSegmention,
    image: np.array,
    prev_coords,
    current_coords,
    thr,
):
    prev = [prev_coords]
    current = [current_coords]

    prev_mask, _ = muti_segmention.inference(inference_image=image, coord_info=prev)

    current_mask, _ = muti_segmention.inference(
        inference_image=image, coord_info=current
    )

    dice = calc_binary_mask_dice(prev_mask, current_mask)

    if dice > thr:
        print(
            f"prev_coords [{prev_coords.coords}] and current_coords [{current_coords.coords}] dice = {dice}, its a same verteba, delete..."
        )
        return True
    return False


def filter_same_coords(
    muti_segmention: VertebraSegmention, image, coords_list, thr=0.3
):
    prev_coord = None
    new_coords = copy.deepcopy(coords_list)
    del_coords = []

    for index, current_coord in enumerate(coords_list):
        if prev_coord is not None:
            print(
                f"\ncheck coords is same, scanning coords list at {index}/{len(coords_list)-1}..."
            )
            if is_same_vertebra(muti_segmention, image, prev_coord, current_coord, thr):
                new_coords.remove(prev_coord)
                del_coords.append(prev_coord)

        prev_coord = current_coord

    return new_coords, del_coords


def filter_valid_coords(
    muti_segmention: VertebraSegmention,
    image,
    coords_list,
    threshold=7820,
    threshold_ratio=0.75,
):
    new_coords = copy.deepcopy(coords_list)
    del_coords = []

    for index, current_coord in enumerate(coords_list):

        print(f"check valid coords {current_coord.coords}, at index {index}...")
        mask, _ = muti_segmention.inference(
            inference_image=image, coord_info=[current_coord]
        )

        count = np.count_nonzero(mask)

        sitk.WriteImage(
            npimage_to_sitk(mask),
            f"test_out/{index}_{current_coord.label}_{count}_{current_coord.coords}.nii.gz",
        )

        if count <= threshold * threshold_ratio:
            print(f"the coords {current_coord.coords} count_nonzero {count}, delete...")
            new_coords.remove(current_coord)
            del_coords.append(current_coord)

    return new_coords, del_coords


def make_labels_continuous(coords):

    if len(coords) <= 1:
        return coords

    coords_info_copy = copy.deepcopy(coords)
    indices = find_max_continuous_coords(coords_info_copy)

    if len(indices) == 0:
        return coords

    if len(indices) != len(coords_info_copy):
        print(f"find_max_continuous_coords_indices = {indices}")
        min_label = coords_info_copy[indices[0]].label
        max_label = coords_info_copy[indices[-1]].label

        for i in reversed(range(0, indices[0])):
            min_label -= 1
            coords_info_copy[i].label = min_label
            # if min_label == 1:
            #     break

        for i in range(indices[-1] + 1, len(coords_info_copy)):
            max_label += 1
            coords_info_copy[i].label = max_label
            # if max_label == 26:
            #     break
    return coords_info_copy


def update_coords_label(
    label_classifier: VertebraLabelClassifier, binary_mask, coords_info
):
    coords_info_copy = copy.deepcopy(coords_info)

    label_info, coords_info_copy = label_classifier.inference(
        binary_mask, coords_info_copy
    )

    # coords_info_copy = sorted(coords_info_copy, key=lambda x: x.coords[2])
    # coords_info_copy = list(dict.fromkeys(coords_info_copy))
    coords_info_copy = make_labels_continuous(coords_info_copy)
    # coords_info_copy = list(dict.fromkeys(coords_info_copy))
    return coords_info_copy


def calc_binary_mask_dice(mask1, mask2):
    bin_mask1 = make_binary_mask(mask1)
    bin_mask2 = make_binary_mask(mask2)
    tp = np.sum(np.logical_and(bin_mask1 == 1, bin_mask2 == 1))
    fp = np.sum(np.logical_and(bin_mask1 == 1, bin_mask2 != 1))
    fn = np.sum(np.logical_and(bin_mask1 != 1, bin_mask2 == 1))
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan
    return dice


def calc_binary_mask_iou(mask1, mask2):
    bin_mask1 = make_binary_mask(mask1)
    bin_mask2 = make_binary_mask(mask2)

    intersection = np.logical_and(bin_mask1, bin_mask2).astype(np.int32)
    union = np.logical_or(bin_mask1, bin_mask2).astype(np.int32)
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    if union_count == 0:
        return 0
    else:
        return intersection_count / union_count


def filter_coords_on_roi(coords, roi_box):
    new_coords = []
    min_z = roi_box.min_coords[2] + 10
    max_z = roi_box.max_coords[2] - 10
    for coord in coords:
        if coord.coords[2] >= min_z and coord.coords[2] <= max_z:
            new_coords.append(coord)
        else:
            print(f"coords closed bottom/top {coord.label}-->{coord.coords}, delete...")
    return new_coords


def contain_undefined_label(coords):
    for coord in coords:
        if coord.label == -1:
            return True
    return False


def is_continuous_label(coords_list, step=1):
    prev = None
    for current in coords_list:
        if prev is not None:
            if step != (current.label - prev.label):
                return False
        prev = current
    return True


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_coords_to_json(json_path, coords_list):
    coords_dict_list = []
    for coords in coords_list:
        coords_dict_list.append(
            {
                "label": float(coords.label),
                "X": float(coords.coords[0]),
                "Y": float(coords.coords[1]),
                "Z": float(coords.coords[2]),
                "value": float(coords.value),
            }
        )
    with open(json_path, "w") as json_file:
        json.dump(coords_dict_list, json_file, indent=4)


def get_roi_box(binary_mask):

    binary_mask_copy = np.copy(binary_mask)
    binary_mask_copy = binary_erosion(binary_mask_copy).astype(binary_mask_copy.dtype)
    x_array, y_array, z_array = np.where(binary_mask_copy > 0)

    x_min, x_max = np.min(x_array), np.max(x_array)
    y_min, y_max = np.min(y_array), np.max(y_array)
    z_min, z_max = np.min(z_array), np.max(z_array)

    roi_box = Box(
        min_coords=np.array([x_min, y_min, z_min]),
        max_coords=np.array([x_max, y_max, z_max]),
    )

    return roi_box

class DiceMetric():
    def __init__(self) -> None:
        pass
    
    def calculate_tp_fp_fn(self, prediction_np, groundtruth_np, label):
        
        if prediction_np.shape != groundtruth_np.shape:
            return 0, 0, 0

        tp = np.sum(np.logical_and(prediction_np ==
                    label, groundtruth_np == label))
        fp = np.sum(np.logical_and(prediction_np ==
                    label, groundtruth_np != label))
        fn = np.sum(np.logical_and(prediction_np !=
                    label, groundtruth_np == label))

        return tp, fp, fn
    
    def __call__(self, prediction_np, groundtruth_np, labels):

        tp_fp_fn_list = [self.calculate_tp_fp_fn(prediction_np, groundtruth_np, label) for label in labels]
        
        dice_list = [self.dice_function(tp, fp, fn) for tp, fp, fn in tp_fp_fn_list ]
        
        return np.nanmean(dice_list)
        
    def dice_function(self, tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan

def inference_single_image(
    inference_image,
    save_path,
    seg_file,
    idv_locate,
    vertebra_segmention,
    label_classifier,
    reprocess=True,
):

    inference_image_name = inference_image

    logs = []
    
    inference_image = read_image(inference_image)    
    print(f"\nprocessing iamge {inference_image_name} image_size {inference_image.shape}...")

    seg_mask = read_image(seg_file)
    binary_mask = make_binary_mask(seg_mask)
    
    roi_box = get_roi_box(binary_mask)
    
    _, coords_info, _ = idv_locate.inference(inference_image=inference_image_name, save_path=None, roi_box=roi_box)
    
    muti_label_mask, _ = vertebra_segmention.inference(inference_image, save_path, coords_info, filter=False)

    muti_label_dice=0
    # muti_label_dice = DiceMetric()(prediction_np=muti_label_mask, groundtruth_np=seg_mask, labels=[i for i in range(1, 25)])
    
    # coords_info = updata_coords_ctd(muti_label_mask)
    
    if reprocess:
        for i in range(5):
            print(
                f"\nmuti_label_dice = {muti_label_dice} , post_processing restart segment stage {i+1}, auto detect miss segment..."
            )
            coords_info_a = auto_resegment_prev(vertebra_segmention, inference_image, binary_mask, muti_label_mask)

            coords_info_b = manual_resegment_prev(roi_box, coords_info + coords_info_a)

            new_coords = sorted(
                coords_info_a + coords_info_b + coords_info,
                key=lambda x: x.coords[2],
                reverse=True,
            )

            new_coords = filter_coords_on_roi(new_coords, roi_box)

            if contain_undefined_label(new_coords) or not is_continuous_label(coords_info):

                print("\nfind miss segment coords, relabel the new coords...")

                # for j in range(3):

                #     print(f"\nrestart segment stage {j+1} on coords...")
                #     if j == 2:
                #         new_save_path = save_path
                #     else:
                #         new_save_path = None

                #       if j in [0, 2]:
                new_coords, del_coords = filter_same_coords(vertebra_segmention, inference_image, new_coords, thr=0.35)
                
                new_coords = update_coords_label(label_classifier, binary_mask, new_coords)

                new_muti_label_mask, _ = vertebra_segmention.inference(inference_image, None, new_coords, filter=False)

                new_coords = updata_coords_ctd(new_muti_label_mask)

                new_muti_label_dice = DiceMetric()(prediction_np=new_muti_label_mask, groundtruth_np=seg_mask, labels=[i for i in range(1, 25)])


                    
                print(f"post_processing muti_label_dice  {new_muti_label_dice} --> {muti_label_dice}, update result...")

                coords_info = new_coords
                muti_label_mask = new_muti_label_mask
                muti_label_dice = new_muti_label_dice
                    


            else:
                print("\nno new coords found, break...\n")
                break

    mutilabel_segmention_logs = f"idv_muti_label_dice = {muti_label_dice}"
    logs.append(mutilabel_segmention_logs)
    print(mutilabel_segmention_logs)


if __name__ == "__main__":

    # idv_seg = VertebraSegmention(
    #     model_func=UNet3D_CB(
    #         in_channels=2,
    #         out_channels=1,
    #         f_maps=64,
    #         layer_order="cbl",
    #         final_activation="sigmoid",
    #         num_levels=5,
    #         use_attn=False,
    #     ),
    #     model_file=r"result\model_weight\test_model\vertebra_unetcb_64_channels\UNet3D_CB_00100.pth",
    #     sigmas=3.0
    # )
    
    idv_seg = VertebraSegmention(
        model_func=UNet3D_CB(in_channels=2, out_channels=1, f_maps=16, layer_order="cbl", final_activation="sigmoid", num_levels=5, use_attn=False),
        model_file=r"result\model_weight\test_model\vertebra_unetcb_96_offset_6.0\last_train_weight.pth",
        sigmas=6.0
    )

    idv_loacte = VertebraLocate_25_Channel(
        model_file=r"result\model_weight\test_model\scn_unet\SCN_UNet_00100.pth",
        model_func=SCN_unet.SCN_UNet(in_channels=1, out_channels=25, f_maps=32),
        min_landmark_value=0.1
    )

    # label_classifier = VertebraLabelClassifier(
    #     model_file={
    #         "group": r"result\model_weight\test_model\classifier\group\ResNet_00080.pth",
    #         "cervical": r"result\model_weight\test_model\classifier\cervical\ResNet_00080.pth",
    #         "thoracic": r"result\model_weight\test_model\classifier\thoracic\ResNet_00080.pth",
    #         "lumbar": r"result\model_weight\test_model\classifier\lumbar\ResNet_00080.pth",
    #     }
    # )

    label_classifier = VertebraLabelClassifier(
        model_func=encoder3d.encoder3d,
        model_file={
            "group": r"result\model_weight\test_model\classifier_encoder3d_SGD\group\encoder3d_00030.pth",
            "cervical": r"result\model_weight\test_model\classifier_encoder3d_SGD\cervical\encoder3d_00090.pth",
            "thoracic": r"result\model_weight\test_model\classifier_encoder3d_SGD\thoracic\encoder3d_00090.pth",
            "lumbar": r"result\model_weight\test_model\classifier_encoder3d_SGD\lumbar\encoder3d_00090.pth",
        }
    )

    # verse598,verse604,verse757,GL279,verse552,verse557,verse612,verse526,verse605
    # inference_image_path = r"test_data\Verse2020\GL428\GL428_CT_ax.nii.gz"
    # seg_path = r"test_data\Verse2020\GL428\GL428_CT_ax_seg.nii.gz"
    # save_path = r"test_out"
    
    # inference_single_image(
    #     inference_image=inference_image_path,
    #     save_path=save_path,
    #     seg_file=seg_path,
    #     idv_locate=idv_loacte,
    #     vertebra_segmention=idv_seg,
    #     label_classifier=label_classifier,
    #     reprocess=True
    # )

    image_folder = r"test_data\Verse2020"
    output_folder = r"idv_segment_locate_postproc\postproc"
    postproc_image_list = []
    check_dir(output_folder)
    ids = os.listdir(image_folder)
    func_args = []
    
    for id in ids:
        filenames = glob(os.path.join(image_folder, id, '*.nii.gz'))
        for filename in sorted(filenames):
            if "_seg" in filename:
                seg_file = filename
            elif "heatmap" not in filename:
                image_file = filename
        output_path = os.path.join(output_folder, id)
        check_dir(output_path)

        inference_single_image(
            inference_image=image_file,
            save_path=output_path,
            seg_file=seg_file,
            idv_locate=idv_loacte,
            vertebra_segmention=idv_seg,
            label_classifier=label_classifier,
            reprocess=False
        )