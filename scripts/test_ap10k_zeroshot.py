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
         model_name = 'hrnet',
         disable_reindexing=False,
         log_path = 'no_log_path_given'
         ):

    set_seed_reproducability(seed=seed)
    get_device(device=device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    flip_test_images = not disable_flip_test_images

    ## Changing data config to be compatible with pretrained in COCO 
    data_cfg = AP10K_configs.AP10K_data_cfg
    data_cfg['image_size'] = [192, 256]
    data_cfg['heatmap_size'] = [48, 64]
    print('Image Resolution: ', data_cfg['image_size'])

    print("\nLoading validation datasets...")
    ds_val = AnimalAP10KDataset(
        f'{AP10K_configs.AP10K_data_root}/annotations/ap10k-val-split1.json',
        f'{AP10K_configs.AP10K_data_root}/data/',
        AP10K_configs.AP10K_data_cfg,
        AP10K_configs.AP10K_val_pipeline,
        dataset_info=AP10K_configs.AP10K_dataset_info,
        test_mode=True)
    
    print('Re indexing: ', not disable_reindexing)
    
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
        re_order_index=not disable_reindexing,
        log_path = os.path.join(log_path, exp_name)
    )
    test.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", default='testing_ap10k_zeroshot',
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, # default=str(datetime.now().strftime("%Y%m%d_%H%M"))
                        )
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
    parser.add_argument("--disable_reindexing", help="disables reindexing of output channels", action="store_true")
    parser.add_argument("--log_path", help="To store coco json results", type=str)
    
    args = parser.parse_args()

    """
    ## Results for Validation set of Ap10k on a COCO trained poseresnet (256x192), without re indexing: 
    
    ## Results for Validation set of Ap10k on a COCO trained poseresnet (256x192), with this re indexing  [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16]: 
    Test: Loss 0.001199 - Accuracy 0.079253
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.001
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.007
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.001
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.002
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.011
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.002
    AP:  OrderedDict([('AP', 0.0010109225208235109), ('AP .5', 0.0065118654722615115), ('AP .75', 0.0), ('AP (M)', 0.0), ('AP (L)', 0.0010163916391639164), ('AR', 0.0018181121560431905), ('AR .5', 0.010829327760362243), ('AR .75', 0.0), ('AR (M)', 0.0), ('AR (L)', 0.001832604909666379)])


    ## Results for Validation set of Ap10k on a COCO trained poseresnet (with adversarial pretraining) (256x192), with this re indexing  [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16]: 
    Test: Loss 0.001194 - Accuracy 0.072974
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.002
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.009
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.002
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.004
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.024
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.004
    AP:  OrderedDict([('AP', 0.0016305202612255914), ('AP .5', 0.008693227128993305), ('AP .75', 0.0), ('AP (M)', 0.0), ('AP (L)', 0.001645588309383791), ('AR', 0.004431502405284832), ('AR .5', 0.023863123926356547), ('AR .75', 0.0), ('AR (M)', 0.0), ('AR (L)', 0.00445558325959452)])

    """
        
    main(**args.__dict__)
