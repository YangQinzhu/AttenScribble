import numpy as np
from medpy import metric


def miccai_calculate_metric_percase(pred, gt, spacing=None):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    res = {
        "dice": np.nan,
        "hd95": np.nan,
        "asd": np.nan,
    }

    if gt.sum() == 0:
        # there is nothing to compare against!
        return res
    if pred.sum() == 0:
        res["dice"] = 0.0
        return res
    res["dice"] = metric.binary.dc(pred, gt)
    res["asd"] = metric.binary.asd(pred, gt, voxelspacing=spacing)
    res["hd95"] = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return res


def miccai_calculate_metric_percase_old(pred, gt, spacing=None):
    """the incorrect but commonly adopted way"""
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    res = {
        "dice": 0.0,
        "hd95": 0.0,
        "asd": 0.0,
    }

    if gt.sum() == 0:
        # there is nothing to compare against!
        return res
    if pred.sum() == 0:
        res["dice"] = 0.0
        return res
    res["dice"] = metric.binary.dc(pred, gt)
    res["asd"] = metric.binary.asd(pred, gt, voxelspacing=spacing)
    res["hd95"] = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return res

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)