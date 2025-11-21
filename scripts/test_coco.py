import argparse
import ast
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(1, os.getcwd())
from datasets.COCO import COCODataset
from training.COCO import COCOTrain
from misc.general_utils import set_seed_reproducability, get_device
from datasets.CustomDS.COCODataset import TopDownCocoDataset
import datasets.CustomDS.data_configs.COCO_configs as COCO_configs

def main(exp_name,
         batch_size=1,
         num_workers=4,
         pretrained_weight_path=None,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         disable_flip_test_images=False,
         seed=1,
         device=None,
         model_name = 'hrnet',
         image_resolution='(256, 192)',
         ):

    # Seeds
    set_seed_reproducability(seed)

    device = get_device(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    flip_test_images = not disable_flip_test_images
    image_resolution = ast.literal_eval(image_resolution)

    print("\nLoading validation datasets...")

    from misc.general_utils import get_coco_loaders
    ds_val = get_coco_loaders(image_resolution=image_resolution, model_name=model_name,
                                phase="val", test_mode=True) # test_mode should not be false here

    from testing.Test import Test
    test = Test(
        ds_test=ds_val, 
        batch_size=batch_size,
        num_workers=num_workers, 
        loss='JointsMSELoss',
        checkpoint_path=None,
        model_c=model_c,
        model_nof_joints=model_nof_joints,
        model_bn_momentum=model_bn_momentum,
        flip_test_images=flip_test_images,
        device=device,
        pre_trained_only = True, 
        pretrained_weight_path = pretrained_weight_path,
        model_name=model_name,
        log_path = f'{exp_name}/test_coco'
    )
    test.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=32)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=4)
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true")
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--model_name", help="poseresnet or hrnet", type=str, default='hrnet')
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(256, 192)')

    args = parser.parse_args()

    
    main(**args.__dict__)
