import argparse
import ast
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(1, os.getcwd())

from datasets.CustomDS.data_configs import AP10K_configs
from misc.general_utils import set_seed_reproducability, get_device
from datasets.CustomDS.AnimalAP10KDataset import AnimalAP10KDataset

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
         model_name = 'hrnet'
         ):

    set_seed_reproducability(seed=seed)
    get_device(device=device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    flip_test_images = not disable_flip_test_images
    image_resolution = AP10K_configs.AP10K_data_cfg['image_size'][::-1]
    print('Image Resolution: ', image_resolution)

    print("\nLoading validation datasets...")
    ds_val = AnimalAP10KDataset(
        f'{AP10K_configs.AP10K_data_root}/annotations/ap10k-val-split1.json',
        f'{AP10K_configs.AP10K_data_root}/data/',
        AP10K_configs.AP10K_data_cfg,
        AP10K_configs.AP10K_val_pipeline,
        dataset_info=AP10K_configs.AP10K_dataset_info,
        test_mode=True)
    
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

    """
    Results for Validation set of Ap10k on poseresnet50 (256x256)
    Test: Loss 0.000332 - Accuracy 0.801165
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.705
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.942
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.781
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.545
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.709
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.738
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.950
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.809
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.548
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.741
    AP:  OrderedDict({'AP': np.float64(0.705187733054311), 'AP .5': np.float64(0.9420956177040277), 'AP .75': np.float64(0.7808868140904278), 'AP (M)': np.float64(0.5447829398324447), 'AP (L)': np.float64(0.7091583798798152), 'AR': np.float64(0.7379255190755988), 'AR .5': np.float64(0.9504095492200171), 'AR .75': np.float64(0.8091246677172157), 'AR (M)': np.float64(0.5484615384615384), 'AR (L)': np.float64(0.7408141930089084)})
    """
        
    main(**args.__dict__)
