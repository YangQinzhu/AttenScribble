
import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm

from networks.unetformer import UNetFormer
from utils.losses import DiceLoss, get_mask_by_radius, cams_to_affinity_label, get_aff_loss
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from utils.attn_gate_crf_loss import pre_compute_xy_mask, AttnGatedCRFV2
from utils.attn_gate_crf_loss import AttnGatedCRF
from miccai_config import gen_config
from dataloaders.data_factory import data_factory

import warnings
warnings.filterwarnings("ignore")


def test(args):

    num_classes = args.num_classes
    model = UNetFormer(
        in_chns=1, 
        class_num=num_classes,
        is_tfm=args.is_tfm,
        is_sep_tfm=args.is_sep_tfm,
    ).cuda()
    db_train, db_val, trainloader, valloader, tester = data_factory(args)

    model.load_state_dict(torch.load(args.test_chpt))
    print("init weight from {}".format(args.test_chpt))
    
    model.eval()
    tester.model = model
    per_class_mean, avg_metric = tester.test_all_volume()

    for c in range(1, num_classes):
        print('testing class %d : test_dice : %f (%f) test_hd95 : %f (%f)' % (c, per_class_mean[c]["dice"], per_class_mean[c]["dice_std"], per_class_mean[c]["hd95"], per_class_mean[c]["hd95_std"]))

    print('testing mean : test_dice : %f (%f) test_hd95 : %f (%f)' % (avg_metric["dice"], avg_metric["dice_std"], avg_metric["hd95"], avg_metric["hd95_std"]))
if __name__ == '__main__':
    args = gen_config()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.test_save_path = os.path.join(
        os.path.dirname(args.test_chpt),
        "{}_pred".format(args.test_chpt.split("/")[-1].replace(".pth", "")),
    )
    if not os.path.exists(args.test_save_path):
        os.makedirs(args.test_save_path, exist_ok=True)
    test(args)
