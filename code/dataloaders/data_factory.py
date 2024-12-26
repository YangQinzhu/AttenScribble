from .miccai_dataset import BaseDataSets, RandomGenerator, collate
from .dataset_chaos import CHAOSBaseDataSets, CHAOSRandomGenerator, CHAOScollate
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.main_eval import Tester

def data_factory(args):
    if args.task == "acdc":
        
        # spacing_json="path_to_/data/ACDC/spacing_info.json"
        spacing_json="./data/ACDC/spacing_info.json"
        ds_class = BaseDataSets
        tfm_class = RandomGenerator
        cl_fun = collate
    elif args.task == "chaos":
        spacing_json="path_to_/data/CHAOS/spacing_info.json"
        ds_class = CHAOSBaseDataSets
        tfm_class = CHAOSRandomGenerator
        cl_fun = CHAOScollate
    else:
        raise NotImplementedError()

    db_train = ds_class(
        base_dir=args.root_path, 
        split="train", 
        transform=transforms.Compose([
            tfm_class(args.patch_size)
        ]), 
        fold=args.fold, 
        rw_portion=args.rw_portion,
        n_rw_per_image=args.n_rw_per_image,
        is_report_rw_metric=args.is_report_rw_metric,
        n_classes=args.num_classes,
    )

    db_val = ds_class(
        base_dir=args.root_path,
        split="val",
        fold=args.fold,
        n_classes=args.num_classes,
    )
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # <mtian> change num_workers if not for debugging purposes
    num_workers = 8
    trainloader = DataLoader(
        db_train, 
        batch_size=args.batch_size, 
        shuffle=True,                
        num_workers=num_workers,
        pin_memory=True, 
        worker_init_fn=worker_init_fn, 
        collate_fn = cl_fun
    )
    valloader = DataLoader(
        db_val, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn = cl_fun
    )
    tester = Tester(
         data_loader=valloader, 
         model=None, 
         n_classes=args.num_classes, 
         save_path=args.test_save_path, 
         spacing_json=spacing_json, 
         is_save=args.is_test_save,
    )
    return db_train, db_val, trainloader, valloader, tester
