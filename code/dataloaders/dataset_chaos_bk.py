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
import SimpleITK as sitk
from skimage import exposure
from torchvision import transforms
from pydicom import dcmread

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):        
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.sup_type = "label"
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(self._base_dir)
            
            print("self.all_slices", self.all_slices)
            print("train_ids", train_ids)
            
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*_scribble.dcm'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)            
            
        elif self.split == 'val':
            """
                return voume        
            """            
            self.all_slices = os.listdir(self._base_dir)
            self.all_volumes_for_val = []
            
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*_gt.dcm'.format(ids), x) != None, self.all_slices))
                new_data_list.sort()
                
                ndl_dcm_gt_volume = []
                ndl_dcm_img_volume = []
                for ndl in new_data_list:
                    

                    ndl_dcm_gt = os.path.join(self._base_dir, ndl)
                    ndl_dcm_img = ndl_dcm_gt.replace("_gt.dcm", ".dcm")
                    ds = dcmread(ndl_dcm_img)
                    print(f"slice location {ds.SliceLocation}")
                    print(f"pixel spacing {ds.PixelSpacing}")
                    print(f"array shape {ds.pixel_array.shape}")
                    
                    ndl_dcm_gt_itk = sitk.ReadImage(ndl_dcm_gt)
                    ndl_dcm_gt_arr = sitk.GetArrayFromImage(ndl_dcm_gt_itk)
                    ndl_dcm_gt_volume.append(ndl_dcm_gt_arr)
                    
                    ndl_dcm_img_itk = sitk.ReadImage(ndl_dcm_img)
                    ndl_dcm_img_arr = sitk.GetArrayFromImage(ndl_dcm_img_itk)
                    
                    ndl_dcm_img_arr = MedicalImageDeal(ndl_dcm_img_arr, percent=0.99).valid_img
                    ndl_dcm_img_arr = (ndl_dcm_img_arr - ndl_dcm_img_arr.min()) / (ndl_dcm_img_arr.max() - ndl_dcm_img_arr.min())
    
                    ndl_dcm_img_volume.append(ndl_dcm_img_arr)
                
                ndl_dcm_gt_volume = np.concatenate(ndl_dcm_gt_volume, axis=0)
                ndl_dcm_img_volume = np.concatenate(ndl_dcm_img_volume, axis=0)
                
                self.sample_list.append([ndl_dcm_gt_volume, ndl_dcm_img_volume])
                import pdb; pdb.set_trace()

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_case_id = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]
        
        all_cases_set = ["patient{:0>2}".format(i) for i in all_case_id]
        fold1_testing_set = [
            "patient{:0>2}".format(i) for i in all_case_id[:7]]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>2}".format(i) for i in all_case_id[7:14]]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>2}".format(i) for i in all_case_id[14:]]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
                       
        if self.split == "train":
            case = self.sample_list[idx]
        
            scribble_path = os.path.join(self._base_dir, case)
            img_path = scribble_path.replace("_scribble.dcm", ".dcm")
            label_path = scribble_path.replace("_scribble.dcm", ".dcm")
            
            img_itk = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img_itk)
            
            img_arr = MedicalImageDeal(img_arr, percent=0.99).valid_img
            img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
            
            img_arr = img_arr[0]
            
            scribble_itk = sitk.ReadImage(scribble_path)
            scribble_arr = sitk.GetArrayFromImage(scribble_itk)
            scribble_arr = scribble_arr[0]
            
            sample = {'image': img_arr, 'label': scribble_arr}
            sample = self.transform(sample)
            
            if self.sup_type == "label":
                label_itk = sitk.ReadImage(label_path)
                label_arr = sitk.GetArrayFromImage(label_itk)
                label_arr = label_arr[0]
                sample = {'image': img_arr, 'label': label_arr}
                sample = self.transform(sample)
        else:
            gt_volume, img_volume = self.sample_list[idx]
            sample = {'image': img_volume, 'label': gt_volume}
            
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label

def random_rot_flip_addPseudoLabel(image, label, pseudo_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    pseudo_label = np.rot90(pseudo_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    pseudo_label = np.flip(pseudo_label, axis=axis).copy()
    return image, label, pseudo_label


def random_rotate_addPseudoLabel(image, label, pseudo_label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    pseudo_label = ndimage.rotate(pseudo_label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label, pseudo_label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if len(sample) == 3:
            image, label, pseudo_label = sample['image'], sample['label'], sample['pseudo_label']
            # ind = random.randrange(0, img.shape[0])
            # image = img[ind, ...]
            # label = lab[ind, ...]
            if random.random() > 0.5:
                image, label, pseudo_label = random_rot_flip_addPseudoLabel(image, label, pseudo_label)
            elif random.random() > 0.5:
                if 4 in np.unique(label):
                    image, label, pseudo_label = random_rotate_addPseudoLabel(image, label, pseudo_label, cval=4)
                else:
                    image, label, pseudo_label = random_rotate_addPseudoLabel(image, label, pseudo_label, cval=0)
            x, y = image.shape
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            pseudo_label = zoom(
                pseudo_label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
            image = torch.from_numpy(
                image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            pseudo_label = torch.from_numpy(pseudo_label.astype(np.uint8))
            
            sample = {'image': image, 'label': label, 'pseudo_label': pseudo_label}
            
            return sample
            
        else:
            image, label = sample['image'], sample['label']
            # ind = random.randrange(0, img.shape[0])
            # image = img[ind, ...]
            # label = lab[ind, ...]
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                if 4 in np.unique(label):
                    image, label = random_rotate(image, label, cval=4)
                else:
                    image, label = random_rotate(image, label, cval=0)
            x, y = image.shape
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(
                image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            sample = {'image': image, 'label': label}
            
            return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

if __name__ == "__main__":
    ds = BaseDataSets(
        base_dir="/data/tianmu/data/BetterScribble/CHAOS/T1_InPhase_TMI_v2",
        split="val",
        transform=transforms.Compose([
            RandomGenerator([256, 256])
        ]), 
        fold="fold1", 
    )