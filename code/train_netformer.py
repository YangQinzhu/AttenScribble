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
from main_config import gen_config
from dataloaders.data_factory import data_factory

import warnings
warnings.filterwarnings("ignore")

def train(args, snapshot_path):
    base_lr = args.base_lr
    attn_lr = args.attn_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = UNetFormer(
        in_chns=1, 
        class_num=num_classes,
        is_tfm=args.is_tfm,
        is_sep_tfm=args.is_sep_tfm,
    ).cuda()
    db_train, db_val, trainloader, valloader, tester = data_factory(args)

    model.train()

    if base_lr == attn_lr:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    else:    
        attn_params = []
        non_attn_params = []
        for name, param in model.named_parameters():
            if 'encoder.block1' in name or "encoder.norm1" in name or "encoder.attn_proj" in name:
                attn_params.append(param)
            else:
                non_attn_params.append(param)
        optimizer = optim.SGD(
            [
                {'params':non_attn_params}, 
                {'params':attn_params}
            ],
            lr=base_lr,
            momentum=0.9, 
            weight_decay=0.0001,
        )
    ce_loss = CrossEntropyLoss(ignore_index=args.ignore_index)
    dice_loss = DiceLoss(num_classes)
    gatecrf_loss = ModelLossSemsegGatedCRF()
    # attn_gatecrf_loss = AttnGatedCRFV2()
    attn_gatecrf_loss = AttnGatedCRF()

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5
    att_loss_mask = get_mask_by_radius(h=64, w=64, radius=8)
    
    # pre-compute the per-location gaussian similarity mask 
    kernel_xy_mask = pre_compute_xy_mask(h=64, w=64, radius=5, sigma=6)

    for epoch_num in range(max_epoch):
        for i_batch, d_batch in enumerate(trainloader):
            volume_batch = d_batch['image'].cuda()
            label_batch = d_batch[args.sup_type].cuda().to(torch.long)
            aux_seg_batch = d_batch[args.aux_seg_type]
            aux_aff_batch = d_batch[args.aux_aff_type]
            
            outputs, attn_pred = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_pred = torch.argmax(outputs_soft, dim=1).detach()
            
            if aux_seg_batch is not None:
                aux_seg_batch = aux_seg_batch.cuda().to(torch.long).detach()
                aux_seg_batch[label_batch != args.ignore_index] = args.ignore_index # any already ground truth scribbled does not participate in rw label
            if aux_aff_batch is not None:
                aux_aff_batch = aux_aff_batch.cuda().to(torch.long).detach()
                aux_aff_batch[label_batch != args.ignore_index] = args.ignore_index # any already ground truth scribbled does not participate in rw label
            # attn_pred will be B, hw, hw
            # generate attn affinity loss
            loss_seg = ce_loss(outputs, label_batch)

            loss_aux_seg = torch.tensor(0.0)
            if args.aux_seg_weight > 0 and aux_seg_batch is not None:
                temp = aux_seg_batch.clone()
                if (args.is_conf_aux_seg) and (iter_num > 1000):
                    temp[outputs_pred != temp] = args.ignore_index # any disagreed location of model pred and rw do not participate in training   
                    if iter_num == 1001:
                        logging.info("conf on aux_seg_batch")
                loss_aux_seg = ce_loss(outputs, temp)
                
            loss_aux_aff = torch.tensor(0.0)
            if args.aux_aff_weight > 0 and attn_pred is not None and aux_aff_batch is not None:
                temp = aux_aff_batch.clone()
                if (args.is_conf_aux_aff) and (iter_num > 1000):
                    temp[outputs_pred != temp] = args.ignore_index # any disagreed location of model pred and rw do not participate in training
                    if iter_num == 1001:
                        logging.info("conf on rw_label_batch_2")
                attn_h, attn_w = 64, 64
                aff_sup = cams_to_affinity_label(temp, attn_h, attn_w, mask=att_loss_mask, ignore_index=args.ignore_index, bg_index=0, ignore_bg_pairs=True)
                loss_aux_aff, aff_pos_count, aff_neg_count = get_aff_loss(attn_pred, aff_sup)

            loss_reg_crf = torch.tensor(0.0)
            if args.reg_crf_weight > 0:
                mask_dst = (label_batch == args.ignore_index).to(torch.float32).unsqueeze(1) # we need not receive energy from scribble labeled locations!
                loss_reg_crf = gatecrf_loss(
                    outputs_soft,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    volume_batch,
                    256,
                    256,
                    mask_src=None, 
                    mask_dst=mask_dst,
                )["loss"]

            # attn gate crf!
            loss_attn_crf = torch.tensor(0.0)
            
            # if args.task == "acdc":
                # cutoff_iter = 1000 # for acdc
            # elif args.task == "chaos":
                # cutoff_iter = 30000 # for chaos
            if args.attn_crf_weight > 0 and attn_pred is not None and iter_num > args.cutoff_iter:
                loss_attn_crf = attn_gatecrf_loss(
                    y_hat_softmax=outputs_soft, 
                    kernel=attn_pred,
                    kernel_xy_mask=kernel_xy_mask, 
                    kernel_h=64, 
                    kernel_w=64,
                    is_y_grad=args.is_y_grad_acrf, 
                    is_k_grad=args.is_k_grad_acrf,
                    is_exclude_sc=args.is_exclude_sc,
                    batch_is_sc=d_batch["is_sc"],
                )["loss"]

            loss = loss_seg + args.reg_crf_weight * loss_reg_crf + args.aux_aff_weight * loss_aux_aff + args.aux_seg_weight * loss_aux_seg + args.attn_crf_weight * loss_attn_crf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if base_lr == attn_lr:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                lr_attn_ = attn_lr * (1.0 - iter_num / max_iterations) ** 0.9
                optimizer.param_groups[0]['lr'] = lr_
                optimizer.param_groups[1]['lr'] = lr_attn_

            iter_num = iter_num + 1

            logging.info(
                'iteration : %d, loss : %f, l_seg : %f, l_crf : %f, l_acrf : %f, l_aff : %f,l_aux_seg : %f, lr : %f, rw_dc : %f, rw_hd : %f, rrw_dc : %f, rrw_hd : %f' % (
                    iter_num, loss.item(), loss_seg.item(), loss_reg_crf.item(), loss_attn_crf.item(), loss_aux_aff.item(), loss_aux_seg.item(), lr_, d_batch["rw_dice"], d_batch["rw_hd"], d_batch["rrw_dice"], d_batch["rrw_hd"]
                ))
            if iter_num > 0 and iter_num % args.eval_interval == 0:
                model.eval()
                tester.model = model
                _, avg_metric = tester.test_all_volume()
                performance = avg_metric["dice"]                                    
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : test_dice : %f (%f) test_hd95 : %f (%f)' % (iter_num, avg_metric["dice"], avg_metric["dice_std"], avg_metric["hd95"], avg_metric["hd95_std"]))
                                
                model.train()            
            if iter_num >= max_iterations:
                logging.info("exceeded max iteration, terminated")
                break
        if iter_num >= max_iterations:
            break
    return "Training Finished!"


if __name__ == "__main__":
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

    snapshot_path = "{}/{}".format(
        args.checkpoint_path, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
