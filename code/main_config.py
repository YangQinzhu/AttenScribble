import argparse

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def gen_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/ACDC', help='Name of Experiment')
    parser.add_argument('--task', type=str, choices=["acdc", "chaos"], default="acdc", help="the task")
    parser.add_argument('--checkpoint_path', type=str, default="", help="root path to save checkpoint")
    parser.add_argument('--exp', type=str,
                        default='ACDC_pCE_GatedCRFLoss', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
    parser.add_argument('--ignore_index', type=int,  default=4, help='the unknown mark index in label map')
    parser.add_argument('--sup_type', type=str, choices=["scribble", "gt", "rw", "rrw"],
                        default='scribble', help='supervision type')
    parser.add_argument('--aux_seg_type', type=str, choices=["scribble", "rw", "rrw"],
                        default='rrw', help='auxilliary supervision type, used to construct seg loss')
    parser.add_argument('--aux_aff_type', type=str, choices=["scribble", "rw", "rrw"],
                        default='rrw', help='auxilliary supervision type, used to construct aff loss')
    # parser.add_argument('--model', type=str, choices=["unetformer", "unetformer_SSA_E1", "unetformer_SSA_E3"],
    #                     default='unetformer', help='model_name')
    parser.add_argument('--model', type=str, 
                        default='unetformer', help='model_name')
    
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    parser.add_argument('--reg_crf_weight', type=float,  default=0.1, help='random seed')
    parser.add_argument('--attn_crf_weight', type=float,  default=0.1, help='attention crf weight')
    parser.add_argument('--aux_aff_weight', type=float,  default=0.1, help='random seed')
    parser.add_argument('--aux_seg_weight', type=float,  default=0.1, help='weight for the rw loss')
    
    parser.add_argument('--rw_portion', type=float,  default=0.7, help='portion of scribble used to generate rw pseudo label')
    parser.add_argument('--n_rw_per_image', type=int,  default=20, help='number of random rw labels to generate per slice')
    parser.add_argument("--is_conf_aux_seg", type=str2bool, nargs='?', const=True, default=False, help="whether to apply conf filtering on rw label when it participate in loss")
    parser.add_argument("--is_conf_aux_aff", type=str2bool, nargs='?', const=True, default=False, help="whether to apply conf filtering on rw label when it participate in aff loss")
    parser.add_argument('--eval_interval', type=float,  default=200, help='weight for the rw loss')
    parser.add_argument("--is_report_rw_metric", type=str2bool, nargs='?', const=True, default=False, help="whether to report rw metric within dataset")

    # modeling specifics
    parser.add_argument("--is_tfm", type=str2bool, nargs='?', const=True, default=False, help="whether or not tfm branch added")
    parser.add_argument("--is_sep_tfm", type=str2bool, nargs='?', const=True, default=False, help="if true, tfm branch is totally seperate, not merging back to main unet stream")

    # attn crf options
    parser.add_argument("--is_y_grad_acrf", type=str2bool, nargs='?', const=True, default=False, help="whether to bp pred logits in attn crf")
    parser.add_argument("--is_k_grad_acrf", type=str2bool, nargs='?', const=True, default=False, help="whether to bp attention induced kernels in attn crf")
    parser.add_argument("--is_exclude_sc", type=str2bool, nargs='?', const=True, default=False, help="whether to exclude single class sample when doing bp in kernels")

    # training control
    parser.add_argument("--cutoff_iter", type=int, default=0, help="cut off iteration number to apply attn crf loss")
    parser.add_argument("--attn_lr", type=float, default=0.01, help="the learning rate specifically for attention blocks")

    # testing control
    parser.add_argument("--test_chpt", type=str, default="", help="path to the checkpoint model to load for testing")
    parser.add_argument("--test_save_path", type=str, default="", help="test save path")
    parser.add_argument("--is_test_save", type=str2bool, nargs='?', const=True, default=False, help="whether to save test or not")

    args = parser.parse_args()
    return args