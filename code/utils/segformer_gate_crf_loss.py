import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class SegformerModelLossSemsegGatedCRF(nn.Module):
    """
    <mtian> adjusted for the segformer model
    we take the output y_hat_softmax, normally should be identical to image resolution, but not necessary so!
    we also take abunch of NCHW feature maps, note H, W, could be different depending on their resolutions; if H, W different to y_hat_softmax, 
    then we should align y_hat_softmax through resizing!        
    we do note that, the automated kernel from attention map could be included as well! This way, there won't be pairwise kernel computation!
    """
    def __init__(self):
        super(SegformerModelLossSemsegGatedCRF, self).__init__()

    def forward(
            self, y_hat_softmax, kernel_configs, feature_inputs, kernels_radius, y_grad=True, feature_grad=True,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False,
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. (for radius, can we do radius in a geodesic way? geodesic map should be pre-computed already)
            kernel_configs: dict<feat_name, config>, e.g. {
                "xy": {
                    "weight": 0.1,
                    "sigma": 6,
                    "is_norm": True,
                },
                "rgb": {
                    "weight": 0.2,
                    "sigma": 6,
                    "is_norm": True,
                },
                "dec1": {},
                "dec2": {},
            }
            <mtian> if only xy involved, then we do not use rgb signal
            <mtian> Note, xy kernel should be replaced by geodesic distance, we could use kernel_radius to bound geodesic distance instead of euclidean distance
            we should also probably normalize each selected feature, so that they sits on a unit sphere;

        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed. This version assumes identical kernels_radius
        :param feature_inputs: dict<feature_name, feature_map>, N, C, H, W; note that H, W does not need to be the same as y_hat_softmax, diff different, we         
        should resize y_hat_softmax to each feature respectively and compute the loss jointly; OR, resize feature to have be same as y_hat_softmax, your call. 
            in this version, feature_inputs includes xy; 
            <mtian> inspired on this, the crf loss does not need to be on full resolution. Geodesic distance can be computed with downsampled image as well.
            e.g. {
                "xy": feature_xy,
                "rgb": feature_rgb,
                "dec1": feature_dec1,
                ...
            }
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask. Must be same H, W, with y_pred
        :param mask_dst: (optional) Destination mask. Must be same H, W, with y_pred
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :param y_grad: whether to backpropagate gradient into y_hat_softmax
        :param feature_grad: whether to backpropagate gradient into features
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        if not y_grad:
            y_hat_softmax = y_hat_softmax.detach()
        if not feature_grad:
            for k, _ in feature_inputs.items():
                feature_inputs[k] = feature_inputs[k].detach()

        N, C, height_pred, width_pred = y_hat_softmax.shape
        device = y_hat_softmax.device

        kernels = self._create_kernels(
            kernel_configs, feature_inputs, kernels_radius, height_pred, width_pred, 
        )

        denom = N * height_pred * width_pred

        def resize_fix_mask(mask, name):
            assert mask.dim() == 4 and mask.shape[:2] == (N, 1) and mask.dtype == torch.float32, \
                f'{name} mask must be a NCHW batch with C=1 and dtype float32'
            if mask.shape[2:] != (height_pred, width_pred):
                mask = resize(mask, size=(height_pred, width_pred), scale_factor=None, mode='bilinear', align_corners=False)
            mask[mask != mask] = 0.0    # handle NaN
            # handle edges of mask after interpolation
            mask[mask < 1.0] = 0.0
            return mask

        if mask_src is not None:
            mask_src = resize_fix_mask(mask_src, 'Source')
            denom = mask_src.sum().clamp(min=1)
            mask_src = self._unfold(mask_src, kernels_radius)
            kernels = kernels * mask_src

        if mask_dst is not None:
            mask_dst = resize_fix_mask(mask_dst, 'Destination')
            denom = mask_dst.sum().clamp(min=1)
            mask_dst = mask_dst.view(N, 1, 1, 1, height_pred, width_pred)
            kernels = kernels * mask_dst

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)

        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)

        if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
            # comment out to save computation, total loss may go below 0
            loss = kernels.sum() + loss
        else:
            assert compatibility.shape == (
                C, C), f'Compatibility matrix expected shape [{C}x{C}]'
            assert (compatibility < 0).int().sum(
            ) == 0, 'Compatibility matrix must not have negative values'
            assert compatibility.diag.sum() == 0, 'Compatibility matrix diagonal must be 0'
            compat = (C-1) * \
                F.normalize(compatibility.float().to(device), p=1, dim=1)
            y_hat_CxNHW = y_hat_softmax.permute(
                1, 0, 2, 3).contiguous().view(C, -1)
            product_kernel_x_y_hat_NHWxC = product_kernel_x_y_hat.permute(
                0, 2, 3, 1).contiguous().view(-1, C)
            product_CxC = torch.mm(y_hat_CxNHW, product_kernel_x_y_hat_NHWxC)
            loss = (compat * product_CxC).sum()

        out = {
            'loss': loss / denom,
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_pred, width_pred, height_pred, width_pred
            )

        return out
    
    @staticmethod
    def _create_kernels(
            kernel_configs, feature_inputs, kernels_radius, height_pred, width_pred,
    ):
        kernels = None
        for feature_name, feature_config in kernel_configs.items():
            weight = feature_config['weight']
            if weight == 0:
                continue
            sigma = feature_config["sigma"]
            is_norm = feature_config["is_norm"]
            feature = feature_inputs[feature_name]
            _, _, hf, wf = feature.size()
            # feature: N, C, H, W
            # resample if needed
            if (hf != height_pred) or (wf != width_pred):
                # we need to resize 
                feature = resize(feature, size=(height_pred, width_pred), scale_factor=None, mode="bilinear", align_corners=False)
            if is_norm:
                # normalize features to unit sphere
                feature = F.normalize(feature, p=2, dim=1)
            feature /= sigma
            kernel = weight * SegformerModelLossSemsegGatedCRF._create_kernels_from_features(feature, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
            # if feature_name not in ["xy", "rgb"]:
                # import pdb; pdb.set_trace()
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        # <mtian> with features, we create gaussian kernels to form a similarity function based on gaussian, note how radius is used here!
        kernels = SegformerModelLossSemsegGatedCRF._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(
            kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3],
                              0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis
