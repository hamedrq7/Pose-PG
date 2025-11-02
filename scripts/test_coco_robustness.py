import argparse
import ast
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(1, os.getcwd())
from misc.general_utils import set_seed_reproducability, get_device
from datasets.CustomDS.COCODataset import TopDownCocoDataset
import datasets.CustomDS.data_configs.COCO_configs as COCO_configs
from datasets.CustomDS.augmentaions import NormalizeTensor

def main(exp_name,
         batch_size=16,
         num_workers=4,
         pretrained_weight_path=None,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         disable_flip_test_images=False,
         seed=1,
         device=None,
         model_name = 'hrnet'
         ):

    # Seeds
    set_seed_reproducability(seed)

    device = get_device(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    flip_test_images = not disable_flip_test_images
    image_resolution = COCO_configs.COCO_data_cfg['image_size'][::-1]
    print('image_resolution', image_resolution)

    print("\nLoading validation datasets...")
    ## Modfiy normalization: the normalization step is added to first on model instead of dataloader
    data_pipeline = COCO_configs.COCO_val_pipeline
    deleted_module = data_pipeline.pop(3)
    
    ## Modifying some other stuff: 
    ds_info = COCO_configs.COCO_dataset_info
    ds_info['joint_weights'] = [
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ]
    # ds_info['joint_weights'] = [
    #     0., # nose
    #     0., # left_eye
    #     0., # right_eye
    #     0., # left_ear
    #     0., # right_ear
    #     0., # left_shoulder
    #     0., # right_shoulder
    #     1., # left_elbow
    #     1., # right_elbow
    #     0., # left_wrist
    #     0., # right_wrist
    #     0., # left_hip
    #     0., # right_hip
    #     0., # left_knee
    #     0., # right_knee
    #     0., # left_ankle
    #     0., # right_ankle
    # ]
    ds_cfg = COCO_configs.COCO_data_cfg
    ds_cfg['use_different_joint_weights'] = True

    assert isinstance(deleted_module, NormalizeTensor), 'Delete the NormalizeTensor module'
    ds_val = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
                        data_cfg=ds_cfg, pipeline=data_pipeline, dataset_info=ds_info)

    from testing.TestRobust import TestRobust
    test = TestRobust(
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
        pretrained_weight_path = pretrained_weight_path,
        model_name=model_name
    )
    test.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=16)
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
    
    args = parser.parse_args()

    
    main(**args.__dict__)
