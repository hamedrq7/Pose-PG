import os
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from misc.checkpoint import load_checkpoint
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from misc.general_utils import get_device, get_model, re_index_model_output, get_loss_fn
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps

import numpy as np 

class Test(object):
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
                 pre_trained_only=False,
                 pretrained_weight_path=None,
                 model_name = 'hrnet',
                 re_order_index = False,
                 log_path = 'no_log_path_given',
                 pck_thresholds = [0.05, 0.2, 0.5]
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
        super(Test, self).__init__()

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

        self.model = get_model(model_name=model_name, model_c=self.model_c, model_nof_joints=self.model_nof_joints,
            model_bn_momentum=self.model_bn_momentum, device=self.device, pretrained_weight_path=pretrained_weight_path)
        if re_order_index: 
            self.model = re_index_model_output(self.model, [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16])

        self.loss_fn = get_loss_fn(self.loss, self.device)
        
        # load test dataset
        self.dl_test = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_test = len(self.dl_test)

        #
        # initialize variables
        self.mean_loss_test = 0.
        self.mean_acc_test = 0.
        self.per_joint_pck_accs = {x: [] for x in self.pck_thresholds}
        self.pck_accs = {x: [] for x in self.pck_thresholds}

    def _test(self):
        num_samples = len(self.ds_test)
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        image_paths = []
        idx = 0
        self.model.eval()

        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_test)):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                output = self.model(image)
                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)

                    output_flipped = flip_back(output_flipped, self.ds_test.flip_pairs)

                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                for pck_thr in self.pck_thresholds[::-1]: 
                    accs, avg_acc, cnt = pose_pck_accuracy(output.detach().cpu().numpy(), 
                                                            target.detach().cpu().numpy(), 
                                                            mask=target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                                                            thr=pck_thr)
                    self.per_joint_pck_accs[pck_thr].append(accs)
                    self.pck_accs[pck_thr].append(avg_acc)
                    
                num_images = image.shape[0]

                # measure elapsed time
                c = joints_data['center'].numpy()
                s = joints_data['scale'].numpy()
                score = joints_data['score'].numpy()
                pixel_std = 200  # ToDo Parametrize this
                bbox_id = joints_data['bbox_id'].numpy()

                preds, maxvals = keypoints_from_heatmaps(
                    heatmaps=output.detach().cpu().numpy(),
                    center=c, 
                    scale=s,
                    post_process="default", 
                    kernel=11, #  if self.ds_test.heatmap_sigma == 2. else 17, # carefull 
                    target_type="GaussianHeatmap",
                )

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2] # .detach().cpu().numpy()
                all_preds[idx:idx + num_images, :, 2:3] = maxvals # .detach().cpu().numpy()
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
                all_boxes[idx:idx + num_images, 5] = score
                all_boxes[idx:idx + num_images, 6] = bbox_id

                image_paths.extend(joints_data['imgPath'])

                idx += num_images

                self.mean_loss_test += loss.item()
                self.mean_acc_test += avg_acc.item()

        self.mean_loss_test /= self.len_dl_test
        self.mean_acc_test /= self.len_dl_test

        for thr in self.pck_thresholds: 
            self.per_joint_pck_accs[thr] = np.array(self.per_joint_pck_accs[thr])
            self.pck_accs[thr] = np.array(self.pck_accs[thr])
            print(f'Per joint PCK acc (thr = {thr}) (Total={self.pck_accs[thr].mean()})')
            for joint in range(self.model_nof_joints):
                temp = self.per_joint_pck_accs[thr][:, joint]
                temp = temp[temp >= 0] # the pck values are negative when joint is not visible
                print(self.ds_test.dataset_info.keypoint_info[joint]['name'], ': ', f'{temp.mean():.4f}', ' | ', end='')
            print()

        print('\nTest: Loss %f - Accuracy %f' % (self.mean_loss_test, self.mean_acc_test))
        print('\nVal AP/AR')
        AP_res = self.ds_test.evaluate( 
            all_preds[:idx], all_boxes[:idx], image_paths[:idx], res_folder=f'{self.log_path}')
        
        print('AP: ', AP_res)
    def run(self):
        """
        Runs the test.
        """
        self.mean_loss_test = 0.
        self.mean_acc_test = 0.

        #
        # Test

        self._test()

        print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
