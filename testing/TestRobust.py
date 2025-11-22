import os
from datetime import datetime

import torch
import torch.nn as nn 
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from misc.checkpoint import load_checkpoint
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from misc.general_utils import NormalizeByChannelMeanStd, get_device, get_model, re_index_model_output, get_loss_fn, perturb
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps
from training.COCO import COCO_standard_epoch_info

import numpy as np 
import sys 
from misc.log_utils import Logger

class TestRobust(object):
    """
    Test class.

    The class provides a basic tool for testing HRNet checkpoints.

    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 ds_test,
                 batch_size=16,
                 num_workers=4,
                 loss='JointsMSELoss',
                 checkpoint_path=None,
                 model_c=48,
                 model_nof_joints=17,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None,
                 pretrained_weight_path=None,
                 model_name = 'hrnet',
                 log_path = 'no_log_path_given',
                 pck_thresholds = [0.05, 0.2, 0.5],
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225],
                 ):
        """
        Initializes a new Test object.

        The HRNet model is initialized and the saved checkpoint is loaded.
        The DataLoader and the loss function are defined.

        Args:
            ds_test (HumanPoseEstimationDataset): test dataset.
            batch_size (int): batch size.
                Default: 1
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            model_c (int): hrnet parameters - number of channels.
                Default: 48
            model_nof_joints (int): hrnet parameters - number of joints.
                Default: 17
            model_bn_momentum (float): hrnet parameters - path to the pretrained weights.
                Default: 0.1
            flip_test_images (bool): flip images during validating.
                Default: True
            device (torch.device): device to be used (default: cuda, if available).
                Default: None
        """
        super(TestRobust, self).__init__()

        self.ds_test = ds_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss = loss
        self.checkpoint_path = checkpoint_path
        self.model_c = model_c
        self.model_nof_joints = model_nof_joints
        self.model_bn_momentum = model_bn_momentum
        self.flip_test_images = flip_test_images
        self.epoch = 0
        self.log_path = log_path
        self.pck_thresholds = pck_thresholds

        self.device = get_device(device)

        os.makedirs(self.log_path, 0o755, exist_ok=True)  # exist_ok=False to avoid overwriting    
        sys.stdout = Logger("{}/run.log".format(log_path))
        # sys.stderr = sys.stdout
        command_line_args = sys.argv
        command = " ".join(command_line_args)
        print(f"The command that ran this script: {command}")

        self.model = get_model(model_name=model_name, model_c=self.model_c, model_nof_joints=self.model_nof_joints,
            model_bn_momentum=self.model_bn_momentum, device=self.device, pretrained_weight_path=pretrained_weight_path)
        # Add the normalizer to the model forward
        normalizer = NormalizeByChannelMeanStd(
            mean=mean, std=std
        ).to(device)
        self.model = nn.Sequential(normalizer, self.model)

        self.loss_fn = get_loss_fn(self.loss, self.device, True)

        # load test dataset
        self.dl_test = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_test = len(self.dl_test)

    def _test(self, epsilon: float, num_steps: int, step_size: float, attack: str = 'pgd'):
        print(f'Running with eps {epsilon}, num_steps {num_steps}, step_size {step_size} attack {attack}')
        # initialize variables
        cln_mean_loss_test = 0.
        cln_mean_acc_test = 0.
        cln_per_joint_pck_accs = {x: [] for x in self.pck_thresholds}
        cln_pck_accs = {x: [] for x in self.pck_thresholds}

        adv_mean_loss_test = 0.
        adv_mean_acc_test = 0.
        adv_per_joint_pck_accs = {x: [] for x in self.pck_thresholds}
        adv_pck_accs = {x: [] for x in self.pck_thresholds}


        cln_epoch_info = COCO_standard_epoch_info(-1, 'cln_test', len(self.ds_test), self.model_nof_joints)
        adv_epoch_info = COCO_standard_epoch_info(-1, 'adv_test', len(self.ds_test), self.model_nof_joints)
        
        self.model.eval()
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_test)):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                ##################################### CLEAN #####################################
                cln_output = self.model(image)
                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    cln_output_flipped = self.model(image_flipped)
                    cln_output_flipped = flip_back(cln_output_flipped, self.ds_test.flip_pairs)
                    cln_output = (cln_output + cln_output_flipped) * 0.5

                cln_loss = self.loss_fn(cln_output, target, target_weight)

                # Evaluate accuracy
                for pck_thr in self.pck_thresholds[::-1]: 
                    cln_accs, cln_avg_acc, cln_cnt = COCO_standard_epoch_info.get_pck_acc(
                        cln_output, 
                        target, 
                        target_weight,
                        pck_thr
                    )
                    cln_per_joint_pck_accs[pck_thr].append(cln_accs)
                    cln_pck_accs[pck_thr].append(cln_avg_acc)

                cln_preds, cln_maxvals = COCO_standard_epoch_info.get_predictions(
                    cln_output, joints_data, use_udp=True if self.model_name == 'vitpose_small' else False
                )

                cln_epoch_info._accumulate_running_stats(cln_loss, cln_accs, cln_avg_acc, cln_cnt)
                cln_epoch_info._accumulate_results_for_mAP(cln_preds, cln_maxvals, joints_data)


                ##################################### ADV #####################################
                adv_image = perturb(self.model, self.device, image, target, target_weight, self.loss_fn, 
                                    epsilon, num_steps, step_size, False)
                adv_output = self.model(adv_image)
                if self.flip_test_images:
                    image_flipped = flip_tensor(adv_image, dim=-1)
                    adv_output_flipped = self.model(image_flipped)
                    adv_output_flipped = flip_back(adv_output_flipped, self.ds_test.flip_pairs)
                    adv_output = (adv_output + adv_output_flipped) * 0.5

                adv_loss = self.loss_fn(adv_output, target, target_weight)

                # Evaluate accuracy
                for pck_thr in self.pck_thresholds[::-1]:
                    adv_accs, adv_avg_acc, adv_cnt = COCO_standard_epoch_info.get_pck_acc(adv_output, target, target_weight, 
                                                                               pck_thr=pck_thr)
                    adv_per_joint_pck_accs[pck_thr].append(adv_accs)
                    adv_pck_accs[pck_thr].append(adv_avg_acc)

                adv_preds, adv_maxvals = COCO_standard_epoch_info.get_predictions(adv_output, joints_data, use_udp=True if self.model_name == "vitpose_small" else False) # You need to link UDP with the part that you create datasets

                adv_epoch_info._accumulate_running_stats(adv_loss, adv_accs, adv_avg_acc, adv_cnt)
                adv_epoch_info._accumulate_results_for_mAP(adv_preds, adv_maxvals, joints_data)


        cln_mean_loss_test = cln_epoch_info.running_loss / self.len_dl_test
        cln_mean_acc_test = cln_epoch_info.running_acc / self.len_dl_test
        adv_mean_loss_test = adv_epoch_info.running_loss / self.len_dl_test
        adv_mean_acc_test = adv_epoch_info.running_acc / self.len_dl_test
        
        print('------', 'CLEAN', '-------')
        for thr in self.pck_thresholds: 
            cln_per_joint_pck_accs[thr] = np.array(cln_per_joint_pck_accs[thr])
            cln_pck_accs[thr] = np.array(cln_pck_accs[thr])
            print(f'Clean per joint PCK acc (thr = {thr}) (Total={cln_pck_accs[thr].mean()})')
            for joint in range(self.model_nof_joints):
                temp = cln_per_joint_pck_accs[thr][:, joint]
                temp = temp[temp >= 0] # the pck values are negative when joint is not visible
                print(self.ds_test.dataset_info.keypoint_info[joint]['name'], ': ', f'{temp.mean():.4f}', ' | ', end='')
            print()

        print('\Clean: Loss %f - Accuracy %f' % (cln_mean_loss_test, cln_mean_acc_test))
        print('\nClean AP/AR')
        AP_res = self.ds_test.evaluate( 
            cln_epoch_info.all_preds[:cln_epoch_info.idx], cln_epoch_info.all_boxes[:cln_epoch_info.idx], cln_epoch_info.image_paths[:cln_epoch_info.idx], res_folder=f'{self.log_path}')
        print('Clean AP: ', AP_res)

        print('------', 'Adv', '-------')
        for thr in self.pck_thresholds: 
            adv_per_joint_pck_accs[thr] = np.array(adv_per_joint_pck_accs[thr])
            adv_pck_accs[thr] = np.array(adv_pck_accs[thr])
            print(f'Adv per joint PCK acc (thr = {thr}) (Total={adv_pck_accs[thr].mean()})')
            for joint in range(self.model_nof_joints):
                temp = adv_per_joint_pck_accs[thr][:, joint]
                temp = temp[temp >= 0] # the pck values are negative when joint is not visible
                print(self.ds_test.dataset_info.keypoint_info[joint]['name'], ': ', f'{temp.mean():.4f}', ' | ', end='')
            print()

        print('\Adv: Loss %f - Accuracy %f' % (adv_mean_loss_test, adv_mean_acc_test))
        print('\nAdv AP/AR')
        AP_res = self.ds_test.evaluate( 
            adv_epoch_info.all_preds[:adv_epoch_info.idx], adv_epoch_info.all_boxes[:adv_epoch_info.idx], adv_epoch_info.image_paths[:adv_epoch_info.idx], res_folder=f'{self.log_path}')
        print('Adv AP: ', AP_res)

    def run(self):
        """
        Runs the test.
        """
        # Test

        self._test(1/255., 20, 0.125/255, 'pgd')
        self._test(2/255., 20, 0.25/255, 'pgd')
        self._test(4/255., 20, 0.5/255, 'pgd')
        self._test(8/255., 20, 1./255, 'pgd')

        print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
