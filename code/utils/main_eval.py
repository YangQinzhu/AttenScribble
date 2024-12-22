"""eval dice and hd95 when physical spacing is given"""

import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import miccai_calculate_metric_percase, miccai_calculate_metric_percase_old

metric_keys = ["dice", "hd95", "asd"]

def _itk_gen_image_from_array(
    image_array, 
    spacing=(0.5, 0.5, 0.5), 
    origin=(0,0,0), 
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
):
    """
    extract left part of hippo in the label
    args:
        image_label: itk image of full hippo label
    return:
        left hippo label, binary mask    
    """

    xxai = sitk.GetImageFromArray(image_array)
    xxai.SetSpacing(spacing)
    xxai.SetOrigin(origin)
    xxai.SetDirection(direction)
    return xxai

def array_to_excel(data, column_name ="proposed", excel_file_path="end"):
    
    import pandas as pd
    # 将NumPy数组转换为pandas DataFrame
    df = pd.DataFrame(data, columns=[column_name])

    # 使用pandas的to_excel方法将DataFrame写入Excel文件
    df.to_excel(excel_file_path, index=False)

    print(f"数据已成功写入: {excel_file_path}")
    
class Tester:
    def __init__(self, data_loader, model, n_classes, save_path, spacing_json, is_save):
        self.data_loader = data_loader
        self.model = model
        self.n_classes = n_classes
        self.all_fg_classes = [x for x in range(1, self.n_classes)]
        self.save_path = save_path
        self.spacing_json = spacing_json
        with open(self.spacing_json, "r") as f:
            self.spacing_info = json.load(f)
        self.is_save = is_save

    def test_single_volume(self, file_path, orig_path, image, prediction, label):
        """image, prediction, label should be numpy array of d, h, w"""
        case = file_path.split("/")[-1].replace(".h5", "")
        spacing = self.spacing_info[case] # this acertains the existence of spacing for this case!
        if spacing is not None:
            spacing_transposed = (spacing[2], spacing[0], spacing[1])
        
        per_class_metrics = {
            k : miccai_calculate_metric_percase_old(prediction == k, label == k, spacing=spacing_transposed) for k in self.all_fg_classes
        }
        
        if self.is_save and self.save_path:

            img_itk = _itk_gen_image_from_array(image, spacing)
            prd_itk = _itk_gen_image_from_array(prediction.astype(np.float32), spacing)
            lab_itk = _itk_gen_image_from_array(label.astype(np.float32), spacing)
            
            sitk.WriteImage(img_itk, os.path.join(self.save_path, case + "_img.nii.gz"))
            sitk.WriteImage(prd_itk, os.path.join(self.save_path, case + "_pred.nii.gz"))
            sitk.WriteImage(lab_itk, os.path.join(self.save_path, case + "_gt.nii.gz"))
            print("saved predict at {}".format(os.path.join(self.save_path, case + "_pred.nii.gz")))
        return per_class_metrics

    def test_all_volume(self):
        # metrics for three classes
        self.model.eval()
        per_item = {}
        per_class_acc = {
            c: {
                "dice": [],
                "dice_std": [],
                "hd95": [],
                "hd95_std": [],
                "asd": [],
                "asd_std": [],
            } for c in self.all_fg_classes
        }
        per_class_mean = {
            c: {
                "dice": np.nan,
                "dice_std": np.nan,
                "hd95": np.nan,
                "hd95_std": np.nan,
                "asd": np.nan,
                "asd_std": np.nan,
            } for c in self.all_fg_classes
        }
        item_id = 0
        
        dice_list_for_cases = []
        hd95_list_for_cases = []
        
        for d_batch in tqdm(self.data_loader):
            # here we take an efficient implementation of the evaluation! we treat the depth dimension as batch dimension, and generate 3D prediction all at once!
            # then feed into metric computation! d_batch should be 1 * 1 * d * h * w
            orig_h, orig_w = d_batch["image"].size(3), d_batch["image"].size(4)
            pred_input = F.interpolate(
                input=d_batch["image"].squeeze(0).squeeze(0).unsqueeze(1),
                size=(256, 256), 
                mode='bilinear',
                align_corners=False,
            ) # d, 1, h, w
            
            pred_output = torch.argmax(
                torch.softmax(
                    self.model(pred_input.cuda())[0].detach(),
                    dim=1,
                ), # d, c, h, w
                dim=1,
            ).unsqueeze(1) # d, 1, h, w
            
           

            """
            change to predict per slices and then combine
            it needs testing
            """

            # if self.model.__class__.__name__ == "UNetFormer_SSA_E1":
            #     # print("pred_input shape", pred_input.shape)
            #     # pred_output = torch.zeros((pred_input.size(0), 1, orig_h, orig_w))
            #     output_torch_to_concate = []
            #     for j in range(pred_input.size(0)):
            #         model_output_j = self.model(pred_input[j].unsqueeze(0).cuda())[0].detach()
            #         output_torch_to_concate.append(model_output_j)
                
            #     pred_output = torch.cat(output_torch_to_concate, dim=0)
                
                
            #     # print("pred_output.shape", pred_output.shape)
            #     pred_output = torch.argmax(
            #         torch.softmax(
            #             pred_output,
            #             dim=1,
            #         ), # d, c, h, w
            #         dim=1,
            #     ).unsqueeze(1)
                
            # else:
            #     pred_output = torch.argmax(
            #         torch.softmax(
            #             self.model(pred_input.cuda())[0].detach(),
            #             dim=1,
            #         ), # d, c, h, w
            #         dim=1,
            #     ).unsqueeze(1)
            
            # move back to original resolution
            pred_output = F.interpolate(
                input=pred_output.to(torch.float32), 
                size=(orig_h, orig_w), 
                mode='nearest',
            ).to(torch.long).squeeze(1).cpu().numpy() # d, orig_h, orig_w
            per_class_metrics = self.test_single_volume(
                file_path=d_batch["path"][0], 
                orig_path=d_batch["orig_path"][0] if d_batch["orig_path"] is not None else None, 
                image=d_batch["image"].detach().squeeze().cpu().numpy(), # h, d, w
                prediction=pred_output, 
                label=d_batch["gt"].squeeze(0).detach().cpu().numpy(), # h, d, w
            )
            
            #  added
            # print("per_class_metrics:", per_class_metrics)
            # print("self.all_fg_classes:", self.all_fg_classes)
            # mean_dice_value = np.nanmean([per_class_metrics[c]["dice"] for c in self.all_fg_classes])
            # print("mean_dice_value:", mean_dice_value)
            # dice_list_for_cases.append(mean_dice_value)
            
            # mean_hd95_value = np.nanmean([per_class_metrics[c]["hd95"] for c in self.all_fg_classes])
            # hd95_list_for_cases.append(mean_hd95_value)
            # print("mean_hd95_value:", mean_hd95_value)
            
            
            for c in self.all_fg_classes:
                for k in metric_keys:
                    per_class_acc[c][k].append(per_class_metrics[c][k])
            per_item[item_id] = {
                k : np.nanmean([per_class_metrics[c][k] for c in self.all_fg_classes]) for k in metric_keys
            }
            item_id += 1
        
        for c in self.all_fg_classes:
            for k in metric_keys:
                per_class_mean[c][k] = np.nanmean(per_class_acc[c][k])                
                per_class_mean[c][k + "_std"] = np.nanstd(per_class_acc[c][k])
        avg_metric = {
            k: np.nanmean([per_class_mean[c][k] for c in self.all_fg_classes]) for k in metric_keys
        }
        avg_metric.update(
            {
                # k + "_std": np.nanmean([per_class_mean[c][k + "_std"] for c in self.all_fg_classes]) for k in metric_keys
                # k + "_std": np.nanstd([per_class_mean[c][k] for c in self.all_fg_classes]) for k in metric_keys 
                k + "_std": np.nanstd([xx[k] for _, xx in per_item.items()]) for k in metric_keys
            }
        )
        
        print("per_class_mean", per_class_mean)
        print("dice_list_for_cases", dice_list_for_cases)
        
        return per_class_mean, avg_metric

if __name__ == "__main__":
    if False:
        Inference(
            fold="fold1",
            root_path="/data/tianmu/data/BetterScribble/WSL4MIS/data/ACDC/ACDC_training_volumes",
            root_raw="/data/tianmu/data/BetterScribble/ACDC/database/training",
            net=None,
            test_save_path=None,
            is_save_pred=False
        )
