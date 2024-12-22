import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from skimage.exposure import rescale_intensity
import tqdm
from skimage.segmentation import random_walker
from multiprocessing import Pool

# def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):    
#     """this is the first version, with very restrict condition, which might not be rational"""
#     if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
#         pseudo_label = np.zeros_like(seed)
#     else:
#         markers = np.ones_like(seed)
#         markers[seed == 4] = 0
#         markers[seed == 0] = 1
#         markers[seed == 1] = 2
#         markers[seed == 2] = 3
#         markers[seed == 3] = 4
#         sigma = 0.35
#         data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                                  out_range=(-1, 1))
#         segmentation = random_walker(data, markers, beta, mode)
#         pseudo_label = segmentation - 1
#     return pseudo_label

def pseudo_label_generator_acdc_v2(data, seed, beta=100, mode='bf', n_classes=4, ignore_index=4):
    """this is the second version, we allow partial seeds to generate random walker!"""
    trigger_rw = False
    for c in range(0, n_classes):
        if (seed == c).max() > 0:
            trigger_rw = True
    if not trigger_rw:
        # we have nothing! return the ignored space
        return np.ones_like(seed) * ignore_index
    # trigger
    markers = np.zeros_like(seed)
    markers[seed == ignore_index] = 0
    for c in range(0, n_classes):
        markers[seed == c] = c + 1    
    sigma = 0.35
    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                out_range=(-1, 1))
    segmentation = random_walker(data, markers, beta, mode)
    pseudo_label = segmentation - 1
    # note that here, we do not have uncertain regions anymore
    return pseudo_label

def _gen_rw(file_path):
    print(file_path)
    h5f = h5py.File(file_path, 'r')
    case_name = file_path.split("/")[-1].split(".h5")[0]
    image = h5f['image'][:]
    label = h5f['label'][:]
    scribble = h5f['scribble'][:]
    # generate deterministic scribble
    rw_label = pseudo_label_generator_acdc_v2(image, scribble)
    f = h5py.File("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices_rw/{}_rw.h5".format(case_name), 'w')
    f.create_dataset('rw_label', data=rw_label, compression="gzip")
    f.close()
    scribble = scribble.reshape(-1)
    # generate rw from randomly stratified sampled scribbles
    idx0 = np.where(scribble == 0)[0]
    idx1 = np.where(scribble == 1)[0]
    idx2 = np.where(scribble == 2)[0]
    idx3 = np.where(scribble == 3)[0]       
    rw_portion = 0.7         
    for i_rw in range(20):
        scribble_i = scribble.copy()
        idx0_drop = np.random.choice(idx0, int(idx0.shape[0] * (1 - rw_portion)), replace=False)
        idx1_drop = np.random.choice(idx1, int(idx1.shape[0] * (1 - rw_portion)), replace=False)
        idx2_drop = np.random.choice(idx2, int(idx2.shape[0] * (1 - rw_portion)), replace=False)
        idx3_drop = np.random.choice(idx3, int(idx3.shape[0] * (1 - rw_portion)), replace=False)
        scribble_i[idx0_drop] = 4
        scribble_i[idx1_drop] = 4
        scribble_i[idx2_drop] = 4
        scribble_i[idx3_drop] = 4
        scribble_i = scribble_i.reshape(label.shape[0], label.shape[1])
        rw_label_i = pseudo_label_generator_acdc_v2(image, scribble_i)
        # write to h5
        f = h5py.File("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices_rw/{}_rw{}_p{}.h5".format(case_name, i_rw, rw_portion), 'w')
        f.create_dataset('rw_label', data=rw_label_i, compression="gzip")
        f.close()

def _get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(1, 21)]
    fold1_training_set = [
        i for i in all_cases_set if i not in fold1_testing_set]
    fold2_testing_set = [
        "patient{:0>3}".format(i) for i in range(21, 41)]
    fold2_training_set = [
        i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
        "patient{:0>3}".format(i) for i in range(41, 61)]
    fold3_training_set = [
        i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
        "patient{:0>3}".format(i) for i in range(61, 81)]
    fold4_training_set = [
        i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
        "patient{:0>3}".format(i) for i in range(81, 101)]
    fold5_training_set = [
        i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    else:
        return "ERROR KEY"

if __name__ == "__main__":
    train_ids, test_ids = _get_fold_ids("fold1")
    all_slices = os.listdir("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices")
    sample_list = []
    for ids in train_ids + test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_slices))
        sample_list.extend(new_data_list)
    all_paths = [
        os.path.join("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices", case) for case in sample_list
    ]
    p = Pool(10)
    p.map(_gen_rw, all_paths)
    # check existence
    if False:
        missing_cases = set()
        for case in sample_list:
            case_name = case.split(".h5")[0]      
            if case_name.endswith("slice_5"):
                case_name_ = case_name.replace("slice_5", "slice_") 
            else:
                case_name_ = case_name

            if not os.path.exists("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices_rw/{}_rw.h5".format(case_name_)):
                # data = h5py.File(os.path.join("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices", case), 'r')
                # import pdb; pdb.set_trace()

                # raise ValueError(f"check {case_name}")
                missing_cases.add(case)
            for i in range(20):
                if not os.path.exists("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices_rw/{}_rw{}_p{}.h5".format(case_name_, i, 0.7)):
                    # raise ValueError(f"check {case_name_}")
                    missing_cases.add(case)
        print(f"cases missing rw")
        print(missing_cases)
        
        missing_cases = [
            'patient081_frame07_slice_15.h5', 
            'patient082_frame01_slice_15.h5', 
            'patient081_frame01_slice_15.h5', 
            'patient096_frame08_slice_15.h5', 
            'patient099_frame01_slice_15.h5', 
            'patient096_frame01_slice_15.h5', 
            'patient088_frame12_slice_15.h5', 
            'patient099_frame09_slice_15.h5', 
            'patient082_frame07_slice_15.h5', 
            'patient088_frame01_slice_15.h5',
        ]
        all_paths = [
            os.path.join("/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_slices", case) for case in missing_cases
        ]
    