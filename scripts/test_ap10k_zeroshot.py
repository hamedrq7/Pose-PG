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
    parser.add_argument("--log_path", help="log dir", type=str)

    args = parser.parse_args()

    """
    ############################################################################################################################################
    # Results for Validation set of Ap10k on a Ap10k trained poseresnet (256x192):
    ############################################################################################################################################
    Per joint PCK acc (thr = 0.05) (Total=0.7588883736089619)
    L_Eye :  0.9464  | R_Eye :  0.9337  | Nose :  0.9218  | Neck :  0.6342  | Root of tail :  0.7101  | L_Shoulder :  0.7271  | L_Elbow :  0.7695  | L_F_Paw :  0.7638  | R_Shoulder :  0.7149  | R_Elbow :  0.7517  | R_F_Paw :  0.7778  | L_Hip :  0.6487  | L_Knee :  0.7373  | L_B_Paw :  0.7313  | R_Hip :  0.6368  | R_Knee :  0.7332  | R_B_Paw :  0.7648  | 
    Per joint PCK acc (thr = 0.2) (Total=0.9588205504921957)
    L_Eye :  0.9845  | R_Eye :  0.9772  | Nose :  0.9702  | Neck :  0.9580  | Root of tail :  0.9497  | L_Shoulder :  0.9789  | L_Elbow :  0.9767  | L_F_Paw :  0.9386  | R_Shoulder :  0.9657  | R_Elbow :  0.9701  | R_F_Paw :  0.9421  | L_Hip :  0.9612  | L_Knee :  0.9679  | L_B_Paw :  0.9272  | R_Hip :  0.9384  | R_Knee :  0.9609  | R_B_Paw :  0.9338  | 
    Per joint PCK acc (thr = 0.5) (Total=0.9957077627999045)
    L_Eye :  0.9970  | R_Eye :  0.9940  | Nose :  0.9928  | Neck :  0.9992  | Root of tail :  0.9858  | L_Shoulder :  0.9989  | L_Elbow :  0.9967  | L_F_Paw :  0.9906  | R_Shoulder :  0.9989  | R_Elbow :  0.9964  | R_F_Paw :  0.9888  | L_Hip :  0.9982  | L_Knee :  0.9979  | L_B_Paw :  0.9986  | R_Hip :  1.0000  | R_Knee :  0.9979  | R_B_Paw :  0.9954  | 

    Test: Loss 0.000419 

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.663
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.920
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.733
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.488
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.667
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.695
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.928
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.767
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.490
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.699
    AP:  OrderedDict({'AP': np.float64(0.6625704485932365), 'AP .5': np.float64(0.9197174888663397), 'AP .75': np.float64(0.7329650645631968), 'AP (M)': np.float64(0.4877253109926377), 'AP (L)': np.float64(0.6670563986606942), 'AR': np.float64(0.6952762716660131), 'AR .5': np.float64(0.9279612579086384), 'AR .75': np.float64(0.766539536062012), 'AR (M)': np.float64(0.49), 'AR (L)': np.float64(0.698524843386261)})
    
    ############################################################################################################################################
    #  Results for Validation set of Ap10k on a COCO trained poseresnet (256x192), with this re indexing  [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16]: 
    ############################################################################################################################################
    Per joint PCK acc (thr = 0.05) (Total=0.0791612754543591)
    L_Eye :  0.1155  | R_Eye :  0.1310  | Nose :  0.1594  | Neck :  0.0529  | Root of tail :  0.0193  | L_Shoulder :  0.0638  | L_Elbow :  0.0541  | L_F_Paw :  0.1225  | R_Shoulder :  0.0786  | R_Elbow :  0.0546  | R_F_Paw :  0.1256  | L_Hip :  0.0474  | L_Knee :  0.0706  | L_B_Paw :  0.0782  | R_Hip :  0.0387  | R_Knee :  0.0668  | R_B_Paw :  0.0645  | 
    Per joint PCK acc (thr = 0.2) (Total=0.4297629924371652)
    L_Eye :  0.6597  | R_Eye :  0.6840  | Nose :  0.5841  | Neck :  0.5001  | Root of tail :  0.2048  | L_Shoulder :  0.3522  | L_Elbow :  0.3522  | L_F_Paw :  0.3332  | R_Shoulder :  0.3934  | R_Elbow :  0.3912  | R_F_Paw :  0.3380  | L_Hip :  0.4326  | L_Knee :  0.4324  | L_B_Paw :  0.3630  | R_Hip :  0.4431  | R_Knee :  0.4718  | R_B_Paw :  0.3686  | 
    Per joint PCK acc (thr = 0.5) (Total=0.8954365478229217)
    L_Eye :  0.9502  | R_Eye :  0.9556  | Nose :  0.9014  | Neck :  0.9522  | Root of tail :  0.8058  | L_Shoulder :  0.8961  | L_Elbow :  0.8643  | L_F_Paw :  0.7858  | R_Shoulder :  0.9225  | R_Elbow :  0.8730  | R_F_Paw :  0.7696  | L_Hip :  0.9387  | L_Knee :  0.9579  | L_B_Paw :  0.8695  | R_Hip :  0.9411  | R_Knee :  0.9506  | R_B_Paw :  0.8911  | 

    Test: Loss 0.001199 

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
    AP:  OrderedDict({'AP': np.float64(0.001012971001533651), 'AP .5': np.float64(0.006532350279362911), 'AP .75': np.float64(0.0), 'AP (M)': np.float64(0.0), 'AP (L)': np.float64(0.0010196919691969196), 'AR': np.float64(0.0018736677115987462), 'AR .5': np.float64(0.011384883315917799), 'AR .75': np.float64(0.0), 'AR (M)': np.float64(0.0), 'AR (L)': np.float64(0.0018932109702724396)})

    ############################################################################################################################################
    # Results for Validation set of Ap10k on a COCO trained poseresnet (with adversarial pretraining) (256x192), with this re indexing  [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16]: 
    ############################################################################################################################################
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
