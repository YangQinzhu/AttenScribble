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
from medpy import metric

ignore_index = 5
n_classes = 5

def CHAOScollate(batch):
    # batch is a list of dict
    res = {
        "idx": [x["idx"] for x in batch],            
        "path": [x["path"] for x in batch] if batch[0]['path'] is not None else None,
        "orig_path": [x["orig_path"] for x in batch] if batch[0]['orig_path'] is not None else None,
        "image": torch.stack([x["image"] for x in batch], 0) if batch[0]['image'] is not None else None,
        "gt": torch.stack([x["gt"] for x in batch], 0) if batch[0]['gt'] is not None else None,
        "rw": torch.stack([x["rw"] for x in batch], 0) if batch[0]['rw'] is not None else None,
        "rrw": torch.stack([x["rrw"] for x in batch], 0) if batch[0]['rrw'] is not None else None,
        "scribble": torch.stack([x["scribble"] for x in batch], 0) if batch[0]['scribble'] is not None else None,
        "rw_dice": np.nanmean([x["rw_dice"] for x in batch]),
        "rw_hd": np.nanmean([x["rw_hd"] for x in batch]),
        "rrw_dice": np.nanmean([x["rrw_dice"] for x in batch]),
        "rrw_hd": np.nanmean([x["rrw_hd"] for x in batch]),       
        'is_sc': [x["is_sc"] for x in batch],
    }
    return res

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0:
        # there is nothing to compare against!
        return np.nan, np.nan
    if pred.sum() == 0:
        return 0, np.nan
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95

def compute_multiclass_metric(pred, gt, n_classes):
    dices = []
    hd95s = []
    for i in range(1, n_classes):
        d, h = calculate_metric_percase(pred == i, gt == i)
        dices.append(d)
        hd95s.append(h)
    return np.nanmean(dices), np.nanmean(hd95s)

def is_valid_label(label, n_classes=n_classes):
    # if label does not include ANY foreground and background, then it is not valid
    for l in range(0, n_classes):
        if (label == l).max() > 0:
            return True
    return False

class CHAOSBaseDataSets(Dataset):
    def __init__(self, 
            base_dir=None, 
            split='train', 
            transform=None, 
            fold="fold1", 
            rw_portion=0.0, 
            n_rw_per_image=20,
            is_report_rw_metric=False,
            n_classes=n_classes,
        ):        
        print(f"CHAOS task triggered, make sure this is correct!")
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.rw_portion = rw_portion
        self.n_rw_per_image = n_rw_per_image
        self.fold = fold
        self.is_report_rw_metric = is_report_rw_metric # whether to compute dice between rw label and true label (as a diagnosis tool)
        self.n_classes = n_classes
        self.ignore_index = n_classes
        self.is_sc = {} # record which sample is of single class
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/CHAOS_training_slices")
            self.sample_list_ = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list_.extend(new_data_list)
            # filter sample_list
            self.sample_list = []
            for case in self.sample_list_:
                path_ = self._base_dir + "/CHAOS_training_slices/{}".format(case)
                h5f = h5py.File(path_, 'r')
                scribble_ = h5f["scribble"][:]
                label_ = h5f["label"][:]
                assert set(np.unique(scribble_[scribble_ != self.ignore_index])) == set(np.unique(label_))

                # if is_valid_label(scribble_) and is_valid_label(label_) and len(set(np.unique(label_))) > 1:
                if is_valid_label(scribble_) and is_valid_label(label_):
                    # this version we remove single class cases!
                    self.sample_list.append(case)
                    self.is_sc[case] = True if len(set(np.unique(label_))) == 1 else False
                else:
                    # print(np.unique(h5f["scribble"][:]), np.unique(h5f["label"][:]))
                    print(f"invalid slice label at {path_}")
                    # print(path_.split("/")[-1].split(".h5")[0])                    
        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/CHAOS_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)
        # stuff in self.sample_list will be patient39_IMG-0027-00036.h5
        print("total {} samples".format(len(self.sample_list)))
    
    def _get_file_lists(self):
        return self.sample_list

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
            raise ValueError(f"{fold} not in [fold1, fold2, fold3]")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
                       
        case = self.sample_list[idx]
        case_name = case.split(".h5")[0]        
        if self.split == "train":
            file_path = self._base_dir + "/CHAOS_training_slices/{}".format(case)
        else:
            file_path = self._base_dir + "/CHAOS_training_volumes/{}".format(case)
        h5f = h5py.File(file_path, 'r')            
        sample = {
            'idx': idx, 
            'path': file_path,
            'orig_path': None,
            'image': None, 
            'scribble': None, 
            'gt': None, 
            'rw': None, 
            'rrw': None,
            'rw_dice': np.nan, # 2d
            'rw_hd': np.nan, # 2d
            'rrw_dice': np.nan, # 2d
            'rrw_hd': np.nan, # 2d
            'is_sc': self.is_sc.get(case, False), # whether this slice ONLY has one single class, regardless of whether it is fg or bg
        } # use label as a placeholder for rw
        if self.split != "train":
            sample["image"] = torch.from_numpy(h5f['image'][:].astype(np.float32)).unsqueeze(0) # expose channel
            sample["gt"] = torch.from_numpy(h5f['label'][:].astype(np.uint8))
            return sample
        # take care of training
        sample["image"] = h5f['image'][:]
        sample["scribble"] = h5f['scribble'][:]
        sample["gt"] = h5f['label'][:]
        if False:
            # we do not yet have rw, rww generated
            h5f_rw = h5py.File(self._base_dir + "/CHAOS_training_slices_rw/{}_rw.h5".format(case_name), 'r')
            sample["rw"] = h5f_rw['rw_label'][:] 
            sample["rw"][(sample["rw"] < 0) | (sample["rw"] > self.n_classes)] = self.ignore_index 
            h5f_rw_i = h5py.File(self._base_dir + "/CHAOS_training_slices_rw/{}_rw{}_p{}.h5".format(case_name, np.random.randint(self.n_rw_per_image), self.rw_portion), 'r')
            sample["rrw"] = h5f_rw_i['rw_label'][:]
            sample["rrw"][(sample["rrw"] < 0) | (sample["rrw"] > self.n_classes)] = self.ignore_index 
            
        sample = self.transform(sample)
        
        # <mtian> h5f has keys image, label, scribble
        # scribble is sparse label mask of the same tensor shape as image, label
        # label: 0, 1, 2, 3 to indicate background and three foreground classes
        # scribble: 0, 1, 2, 3, 4 where 4 indicates "unknown", this index is ignored in loss computation
        if self.is_report_rw_metric:
            if sample['rw'] is not None:
                sample['rw_dice'], sample['rw_hd'] = compute_multiclass_metric(
                    sample['rw'].numpy(),
                    sample['gt'].numpy(),
                    n_classes=self.n_classes,
                )
            if sample['rrw'] is not None:
                sample['rrw_dice'], sample['rrw_hd'] = compute_multiclass_metric(
                    sample['rrw'].numpy(),
                    sample['gt'].numpy(),
                    n_classes=self.n_classes,
                )
        return sample

def random_rot_flip(sample, keys):
    k = np.random.randint(0, 4)
    for lk in keys:
        # sample[lk] = np.rot90(sample[lk], k)
        # disable rotation90 for this dataset
        pass
    axis = np.random.randint(0, 2)
    for lk in keys:
        # sample[lk] = np.flip(sample[lk], axis=axis).copy()
        # sample[lk] = np.flip(sample[lk], axis=axis)
        # disable flipping for this dataset
        pass
    return sample

def random_rotate(sample, keys):
    angle = np.random.randint(-20, 20)
    for lk in keys:
        if lk == "image":
            sample[lk] = ndimage.rotate(sample[lk], angle, order=0, reshape=False)
        else:
            if (sample[lk] == ignore_index).max() > 0:
                sample[lk] = ndimage.rotate(sample[lk], angle, order=0, reshape=False, mode="constant", cval=ignore_index)
            else:
                sample[lk] = ndimage.rotate(sample[lk], angle, order=0, reshape=False, mode="constant", cval=0)
    return sample


class CHAOSRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        # self.keys = ["image", "scribble", "gt", "rw", "rrw"]
        self.keys = ["image", "scribble", "gt"]

    def __call__(self, sample):
        if random.random() > 0.5:
            sample = random_rot_flip(sample, self.keys)
        elif random.random() > 0.5:
            sample = random_rotate(sample, self.keys)
        hx, hy = sample["image"].shape
        for lk in self.keys:
            sample[lk] = zoom(sample[lk], (self.output_size[0] / hx, self.output_size[1] / hy), order=0)
        for lk in self.keys:
            if lk == "image":
                sample[lk] = torch.from_numpy(sample[lk].astype(np.float32)).unsqueeze(0)
            else:
                sample[lk] = torch.from_numpy(sample[lk].astype(np.uint8))
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
    ds = CHAOSBaseDataSets(
        base_dir="/data/tianmu/data/BetterScribble/WSL4MIS/data/CHAOS",
        split="val",
        transform=transforms.Compose([
            CHAOSRandomGenerator([256, 256])
        ]), 
        fold="fold1", 
        n_classes=n_classes,
    )

