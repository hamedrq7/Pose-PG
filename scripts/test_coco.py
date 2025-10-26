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


def main(exp_name,
         batch_size=1,
         num_workers=4,
         pretrained_weight_path=None,
         checkpoint_path=None,
         log_path='./logs',
         disable_tensorboard_log=False,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         disable_flip_test_images=False,
         image_resolution='(384, 288)',
         coco_root_path="./datasets/COCO",
         coco_bbox_path=None,
         seed=1,
         device=None,
         pre_trained_only=True,
         model_name = 'hrnet'
         ):

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True  # Enables cudnn
        torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

    # torch device
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    use_tensorboard = not disable_tensorboard_log
    flip_test_images = not disable_flip_test_images
    image_resolution = ast.literal_eval(image_resolution)

    print("\nLoading validation datasets...")

    # ds_val = COCODataset(
    #     root_path=coco_root_path, data_version="val2017", is_train=False, use_gt_bboxes=(coco_bbox_path is None),
    #     bbox_path=coco_bbox_path, image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True,
    # )

    from datasets.CustomDS.COCODataset import TopDownCocoDataset
    import datasets.CustomDS.COCO_configs as COCO_configs

    ds_val = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
                        data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_val_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

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
        model_name=model_name
    )
    test.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    # parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=200)
    # parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=16)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=4)
    # parser.add_argument("--lr", "-l", help="initial learning rate", type=float, default=0.001)
    # parser.add_argument("--disable_lr_decay", help="disable learning rate decay", action="store_true")
    # parser.add_argument("--lr_decay_steps", help="learning rate decay steps", type=str, default="(170, 200)")
    # parser.add_argument("--lr_decay_gamma", help="learning rate decay gamma", type=float, default=0.1)
    # parser.add_argument("--optimizer", "-o", help="optimizer name. Currently, only `SGD` and `Adam` are supported.",
    #                     type=str, default='Adam')
    # parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.)
    # parser.add_argument("--momentum", "-m", help="momentum", type=float, default=0.9)
    # parser.add_argument("--nesterov", help="enable nesterov", action="store_true")
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--checkpoint_path", "-c",
                        help="previous checkpoint path. Checkpoint will be loaded before training starts. It includes "
                             "the model, the optimizer, the epoch, and other parameters.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./logs')
    parser.add_argument("--disable_tensorboard_log", "-u", help="disable tensorboard logging", action="store_true")
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--coco_root_path", help="COCO dataset root path", type=str, default="./datasets/COCO")
    parser.add_argument("--coco_bbox_path", help="path of detected bboxes to use during evaluation",
                        type=str, default=None)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--model_name", help="poseresnet or hrnet", type=str, default='hrnet')
    
    args = parser.parse_args()

        
    import datasets.CustomDS.COCO_configs as COCO_configs
    from datasets.CustomDS.COCODataset import TopDownCocoDataset
    from torch.utils.data import DataLoader

    
    print('----train')
    val_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/train2017/', 
                        data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_train_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

    val_dl = DataLoader(val_ds, batch_size=2)

    for image, target, target_weight, joints_data in val_dl:
        print(image.shape)
        print(target.shape)
        print(target_weight.shape)
        print(joints_data.keys())

        break


    print('----val')
    val_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
                            data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_val_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

    val_dl = DataLoader(val_ds, batch_size=2)

    for image, target, target_weight, joints_data in val_dl:
        print(image.shape)
        print(target.shape)
        print(target_weight.shape)
        print(joints_data.keys())

        break
        
    main(**args.__dict__)
