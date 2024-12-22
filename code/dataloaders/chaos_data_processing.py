# save images in slice level
import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import re
from pydicom import dcmread
import json

def is_valid_label(label):
    """label unique within 0, 1, 2, 3, 4"""
    return set(np.unique(label)) == set([0, 1, 2, 3, 4])

def is_valid_scribble(scribble):
    """scribble unique within 0, 1, 2, 3, 4, 5"""
    return set(np.unique(scribble)) == set([0, 1, 2, 3, 4, 5])

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

# saving images in slice level
def preprocess():
    patient_to_slices = {}    
    spacing_info = {}
    _base_dir = "/data/tianmu/data/BetterScribble/CHAOS/T1_InPhase_TMI_v2"
    _target_dir = "/data/tianmu/data/BetterScribble/WSL4MIS/data/CHAOS" # where to store processed h5
    all_case_id = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]     
    all_cases_set = ["patient{:0>2}".format(i) for i in all_case_id]
    # stores all patient names
    all_slices = os.listdir(_base_dir)
    for patient_key in all_cases_set:
        print(f"patient_key : {patient_key}")
        new_data_list = list(filter(lambda x: re.match(
            '{}.*_scribble.dcm'.format(patient_key), x) != None, all_slices))
        patient_to_slices[patient_key] = [x.replace("_scribble.dcm", "") for x in new_data_list]
        spacing_info[patient_key] = None

    # order slices in a correct way while collecting spacing info!
    for patient_key, slices in patient_to_slices.items():
        correct_order = [] # (case, slicelocation<sorting key>)
        for case in slices:
            print(f"patient_key : {patient_key}, case : {case}")
            img_path = os.path.join(_base_dir, case + ".dcm")
            ds = dcmread(img_path)
            correct_order.append((case, ds.SliceLocation))
            # push spacing info
            if spacing_info[patient_key] is None:
                spacing_info[patient_key] = (ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness)
        # correct order
        correct_order = sorted(correct_order, key=lambda s: s[1])
        patient_to_slices[patient_key] = [x[0] for x in correct_order]

    # create jsons
    json_path = "/data/tianmu/data/BetterScribble/CHAOS/spacing_info.json"
    with open(json_path, "w") as f:
        json.dump(spacing_info, f, indent=4)
    json_path = "/data/tianmu/data/BetterScribble/CHAOS/patient_to_slices.json"
    with open(json_path, "w") as f:
        json.dump(patient_to_slices, f, indent=4)

    # create h5: slice and volume!
    n_modified_slices = 0
    for patient_key, slices in patient_to_slices.items():
        ds_0 = dcmread(os.path.join(_base_dir, slices[0] + ".dcm"))
        img_shape = (len(slices), ds_0.pixel_array.shape[0], ds_0.pixel_array.shape[1]) # d, h, w
        img3d = np.zeros(img_shape)
        scribble3d = np.zeros(img_shape)
        label3d = np.zeros(img_shape)
        for i, case in enumerate(slices):
            print(f"patient_key : {patient_key}, case : {case}")
            img2d = dcmread(os.path.join(_base_dir, case + ".dcm")).pixel_array
            scribble2d = dcmread(os.path.join(_base_dir, case + "_scribble.dcm")).pixel_array
            label2d = dcmread(os.path.join(_base_dir, case + "_gt.dcm")).pixel_array
            
            if set(np.unique(label2d)) == set([0]) and set(np.unique(scribble2d)) == set([5]):
                n_modified_slices += 1
                print(f"add background scribble here! {n_modified_slices}")
                scribble2d[img_shape[1] // 3, :] = 0
                scribble2d[img_shape[1] * 2 // 3, :] = 0
                scribble2d[:, img_shape[2] // 3] = 0
                scribble2d[:, img_shape[2] * 2 // 3] = 0

            img3d[i, :, :] = img2d
            scribble3d[i, :, :] = scribble2d
            label3d[i, :, :] = label2d

        # pre-process image (do Note that the image to be processed is on the 3D image volume as a whole, NOT individual 2D images!)
        img3d = MedicalImageDeal(img3d, percent=0.99).valid_img
        if img3d.max() == img3d.min():
            img3d = img3d - img3d.min()
        else:
            img3d = (img3d - img3d.min()) / (img3d.max() - img3d.min())
        img3d = img3d.astype(np.float32)
        label3d = label3d.astype(np.int32)
        scribble3d = scribble3d.astype(np.int32)
        assert is_valid_label(label3d)
        assert is_valid_scribble(scribble3d)
        # for scribble: if this is a all-background image, then we should create scribbles as well!
    
        # save volume to h5
        f = h5py.File(os.path.join(_target_dir, "CHAOS_training_volumes", "{}.h5".format(patient_key)), 'w')
        f.create_dataset('image', data=img3d, compression="gzip")
        f.create_dataset('label', data=label3d, compression="gzip")
        f.create_dataset('scribble', data=scribble3d, compression="gzip")
        f.close()

        # save slices to h5
        for i, case in enumerate(slices):
            f = h5py.File(os.path.join(_target_dir, "CHAOS_training_slices", "{}.h5".format(case)), 'w')
            f.create_dataset('image', data=img3d[i, :, :], compression="gzip")
            f.create_dataset('label', data=label3d[i, :, :], compression="gzip")
            f.create_dataset('scribble', data=scribble3d[i, :, :], compression="gzip")
            f.close()

if __name__ == "__main__":
    preprocess()