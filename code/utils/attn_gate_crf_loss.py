import torch
import torch.nn.functional as F
import numpy as np


def gau_sim(p1, p2, sigma):
    """gaussian similarity, the larger, the closer two points be
        p1, p2 are (x, y)
    """
    if p1 == p2:
        return 0.0
    x1, y1 = p1
    x2, y2 = p2
    gau = -0.5 * (1 / (sigma ** 2)) * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
    gau = np.exp(gau)
    return gau

def pre_compute_xy_mask(h, w, radius, sigma):
    """
    compute xy mask, size hw, hw
    """
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w
        # i is the pivot point now; _h, _w are corresponding row and col
        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                # (i1, i2) are neighboring locations
                _i2 = i1 * w + i2
                # _i2 is the flattened neighbor location!
                dc = gau_sim(p1=(_h, _w), p2=(i1, i2), sigma=sigma)
                mask[i, _i2] = dc
                mask[_i2, i] = dc
    return mask


class AttnGatedCRF(torch.nn.Module):
    """
    the kernel (similarity) is provided as B, HW, HW already, we just need to multiply them by predicted logits (pairwise)
    """

    def forward(self, y_hat_softmax, kernel, kernel_xy_mask, kernel_h, kernel_w, is_y_grad=True, is_k_grad=True, is_exclude_sc=False, batch_is_sc=None):
        """
            y_hat_softmax, B, C, H, W
            kernel: B, kernel_h * kernel_w, kernel_h * kernel_w, 
            kernel_xy_mask: HW, HW, pre-computed euclidean distance regulation mask! This is pre-computed based on kernel_h, kernel_w only, might be time consuming, do not compute every time!
            kernel_h, kernel_w: feature map resolution used to construct kernel
            normally, kernel is given by a global attention map from transformer (self-attention)
            is_y_grad: whether to grad on y_hat_softmax
            is_k_grad: whether to grad on kernel
            is_exclude_sc: if the sample in the batch is ONLY single class, then we disable the loss on this sample if is_k_grad
            batch_is_sc: List[] of length B, for each sample in the batch, whether it is single classed or not
        """
        if (not is_y_grad) and (not is_k_grad):
            raise ValueError()
        N, C, H, W = y_hat_softmax.shape
        device = y_hat_softmax.device
        if not is_y_grad:
            y_hat_softmax = y_hat_softmax.detach()
        if not is_k_grad:
            kernel = kernel.detach()
        # take care of single class
        if is_k_grad and is_exclude_sc and (batch_is_sc is not None):
            kernel = kernel * ((~torch.tensor(batch_is_sc)).unsqueeze(1).unsqueeze(1).to(device))
        
        # resize y_hat to the size of kernel!
        if (H != kernel_h) or (W != kernel_w):
            y_hat_softmax = F.interpolate(y_hat_softmax, size=(kernel_h, kernel_w), mode="bilinear", align_corners=False)
                
        mask_self_loop = torch.eye(kernel_h * kernel_w).unsqueeze(0).to(device).to(torch.float32) # mask out attention of self to self! There is no contrast this way
        # first bmm -> N, hw, c
        kernel_xy_mask = torch.from_numpy(kernel_xy_mask).unsqueeze(0).to(device).to(torch.float32)
        # denom = (kernel_h * kernel_w - 1) * (kernel_h * kernel_w) * C * N
        denom = kernel_h * kernel_w * N
        y_hat_softmax = y_hat_softmax.flatten(2).transpose(1, 2).contiguous().to(torch.float32) # N, hw, c
        left_part = (1.0 - mask_self_loop) * kernel * kernel_xy_mask
        right_part = torch.bmm(left_part, y_hat_softmax) * y_hat_softmax # N, hw, c

        energy = (left_part.sum() - right_part.sum()) / denom
        
        out = {
            'loss': energy,
        }
        return out

class AttnGatedCRFV2(torch.nn.Module):
    """
    the kernel (similarity) is provided as B, HW, HW already, we just need to multiply them by predicted logits (pairwise)
    The difference comparing to original version!
    Case 1:
        IF single class:
            <Note if it is single class, we could use gt directly as gt is single class anyway! This is called legal leakage>
            Then loss = B * H * W * H * W - kernel.sum() + B * H * W - (y_hat_softmax * y_hat_softmax).sum() # we want to maximize kernel (maximize similarity among all pairs!); also minimize the chance logits differs
            Another option is simply don't let single class sample participate into this! Since we have full-sup in this case anyway.
        ELSE:
            Then, 
            loss = loss_k + loss_y
            loss_k is a contrastive one: loss_k = kernel * (y disagree).detach() + (1 - kernel) * (y agree).detach()
            loss_y is a traditional one (since the softmax-ness in y already has its regulation) loss_y = kernel.detach() * (y disagree)
        **we here implement the log version to increase stability**
        log seem to create nan
    """

    def forward(self, y_hat_softmax, kernel, kernel_xy_mask, kernel_h, kernel_w, is_y_grad=True, is_k_grad=True, is_exclude_sc=True, batch_is_sc=None):
        """
            y_hat_softmax, B, C, H, W
            kernel: B, kernel_h * kernel_w, kernel_h * kernel_w, 
            kernel_xy_mask: HW, HW, pre-computed euclidean distance regulation mask! This is pre-computed based on kernel_h, kernel_w only, might be time consuming, do not compute every time!
            kernel_h, kernel_w: feature map resolution used to construct kernel
            normally, kernel is given by a global attention map from transformer (self-attention)
            is_y_grad: whether to grad on y_hat_softmax
            is_k_grad: whether to grad on kernel
            batch_is_sc: List[] of length B, for each sample in the batch, whether it is single classed or not
        """
        if (not is_y_grad) and (not is_k_grad):
            raise ValueError()
        N, C, H, W = y_hat_softmax.shape
        device = y_hat_softmax.device
        if not is_y_grad:
            y_hat_softmax = y_hat_softmax.detach()
        if not is_k_grad:
            kernel = kernel.detach()
        
        
        # resize y_hat to the size of kernel!
        if (H != kernel_h) or (W != kernel_w):
            y_hat_softmax = F.interpolate(y_hat_softmax, size=(kernel_h, kernel_w), mode="bilinear", align_corners=False)
        # detached version to construct the loss
        # y_detach = y_hat_softmax.detach()
        # k_detach = kernel.detach()

        mask_self_loop = 1.0 - torch.eye(kernel_h * kernel_w).unsqueeze(0).to(device).to(torch.float32) # mask out attention of self to self! There is no contrast this way
        kernel_xy_mask = torch.from_numpy(kernel_xy_mask).unsqueeze(0).to(device).to(torch.float32) # do not use too-far away points
        # generate loss for single class
        if batch_is_sc is not None:
            mask_is_sc = torch.tensor(batch_is_sc).unsqueeze(1).unsqueeze(1).to(device) # N, 1, 1
        else:
            mask_is_sc = torch.zeros(N, 1, 1).to(torch.bool).to(device)

        loss_for_sc = torch.tensor(0.0)        
        if (batch_is_sc is not None) and mask_is_sc.max().item():
            # loss_for_sc = mask_is_sc.sum() * kernel_h * kernel_w * kernel_h * kernel_w - (mask_self_loop * kernel * mask_is_sc).sum() + mask_is_sc.sum() * kernel_h * kernel_w - (mask_is_sc.unsqueeze(-1) * y_hat_softmax * y_hat_softmax).sum()
            # loss_for_sc =  - (mask_self_loop * torch.log(kernel) * mask_is_sc).mean() - (mask_is_sc.unsqueeze(-1) * torch.log(y_hat_softmax) * y_hat_softmax).mean() 
            # loss_for_mc = loss_mc_k + loss_mc_y       
            # loss_for_sc = loss_for_sc / (mask_is_sc.sum() * kernel_h * kernel_w * kernel_h * kernel_w)
            pass
        if mask_is_sc.min().item():
            # everything in the batch is single classed! then no case 2 involved here
            out = {
                'loss': loss_for_sc,
            }
            return out

        denom = kernel_h * kernel_w * (~mask_is_sc).sum()
        y_k = (y_hat_softmax * (~mask_is_sc.unsqueeze(-1))).flatten(2).transpose(1, 2).contiguous().to(torch.float32) # N, hw, c
        
        # left_part_pos = mask_self_loop * (-torch.log(1.0 - kernel)) * kernel_xy_mask * (~mask_is_sc)
        left_part_pos = mask_self_loop * kernel * kernel_xy_mask * (~mask_is_sc)
        right_part_pos = torch.bmm(left_part_pos, y_k) * y_k # N, hw, c
        loss_mc_k_pos = left_part_pos.sum() - right_part_pos.sum()

        # left_part_neg = mask_self_loop * (-torch.log(kernel)) * kernel_xy_mask * (~mask_is_sc)
        left_part_neg = mask_self_loop * (1.0 - kernel) * kernel_xy_mask * (~mask_is_sc)
        loss_mc_k_neg = (torch.bmm(left_part_neg, y_k) * y_k).sum()

        loss_mc_k = (loss_mc_k_pos + loss_mc_k_neg) / denom

        # y_y = y_hat_softmax.flatten(2).transpose(1, 2).contiguous().to(torch.float32) # N, hw, c
        left_part_y = mask_self_loop * kernel * kernel_xy_mask * (~mask_is_sc)
        right_part_y = torch.bmm(left_part_y, y_k) * y_k # N, hw, c
        loss_mc_y = (left_part_y.sum() - right_part_y.sum()) / denom
        out = {
            # 'loss': loss_for_sc + loss_mc_k + loss_mc_y,
            'loss': loss_mc_k,
        }

        # print(f"debug acrf: loss_for_sc : {loss_for_sc}, loss_mc_k : {loss_mc_k}, loss_mc_y : {loss_mc_y}")
        return out