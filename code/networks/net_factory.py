from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_CCT, UNet_CCT_3H
from networks.unetformer import UNetFormer
from mmseg.models.segmentors.wsl_encoder_decoder import WSLEncoderDecoder
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.decode_heads.wsl_segformer_head import WSLSegformerHead
from mmcv.utils.config import ConfigDict

def net_factory(net_type="unet", in_chns=1, class_num=3, args=None):
    """args: params passed for segformer, etc"""
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnformer":
        # use nnformer as the backbone
        raise NotImplementedError()
    elif net_type == "unetformer":
        net = UNetFormer(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "segformer":
        # the nvidia segformer for semantic segmentation
        # <mtian> we borrow segformer
        net = WSLEncoderDecoder(
            backbone=ConfigDict(
                {'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'drop_rate': 0.0,
                'embed_dims': 64,
                'in_channels': 1,
                'mlp_ratio': 4,
                'num_heads': [1, 2, 5, 8],
                'num_layers': [3, 4, 18, 3],
                'num_stages': 4,
                'out_indices': (0, 1, 2, 3),
                'patch_sizes': [7, 3, 3, 3],
                'qkv_bias': True,
                'sr_ratios': [8, 4, 2, 1],
                'type': 'MixVisionTransformer'}
            ),
            decode_head=ConfigDict(
                {'align_corners': False,
                'channels': 256,
                'dropout_ratio': 0.1,
                'in_channels': [64, 128, 320, 512],
                'in_index': [0, 1, 2, 3],
                'loss_decode': {'loss_weight': 1.0,
                                'type': 'CrossEntropyLoss',
                                'use_sigmoid': False},
                # 'norm_cfg': {'requires_grad': True, 'type': 'BN'},
                'norm_cfg': {'requires_grad': True, 'type': 'LN'},
                'num_classes': 4,
                'type': 'WSLSegformerHead'}
            ),
            neck=None,
            auxiliary_head=None,
            train_cfg=ConfigDict(),
            test_cfg=ConfigDict(),
            pretrained=None,
            init_cfg=None
        ).cuda()
    else:
        net = None
    return net
