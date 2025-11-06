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
from misc.general_utils import NormalizeByChannelMeanStd, get_device, get_model, re_index_model_output, get_loss_fn
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps

import numpy as np 
from torch.autograd import Variable
import torch.optim as optim

def perturb(model, device, X, y, y_t, loss_fn, epsilon, num_steps, step_size, rand_init=False):
    X, y, y_t = Variable(X, requires_grad=True), Variable(y), Variable(y_t)
    X_pgd = Variable(X.data, requires_grad=True)
    
    if rand_init:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    
    for k in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            adv_output = model(X_pgd)   
            adv_loss = loss_fn(adv_output, y, y_t)

        adv_loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd

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
                 re_order_index = False,
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

        self.model = get_model(model_name=model_name, model_c=self.model_c, model_nof_joints=self.model_nof_joints,
            model_bn_momentum=self.model_bn_momentum, device=self.device, pretrained_weight_path=pretrained_weight_path)
        # Add the normalizer to the model forward
        normalizer = NormalizeByChannelMeanStd(
            mean=mean, std=std
        ).to(device)
        self.model = nn.Sequential(normalizer, self.model)

        if re_order_index: 
            self.model = re_index_model_output(self.model, [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16])

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

        num_samples = len(self.ds_test)
        
        cln_all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        cln_all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        cln_image_paths = []
        
        adv_all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        adv_all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        adv_image_paths = []
        
        
        idx = 0
        self.model.eval()
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_test)):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)
                num_images = image.shape[0]

                # measure elapsed time
                c = joints_data['center'].numpy()
                s = joints_data['scale'].numpy()
                score = joints_data['score'].numpy()
                pixel_std = 200  # ToDo Parametrize this
                bbox_id = joints_data['bbox_id'].numpy()

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
                    cln_accs, cln_avg_acc, cln_cnt = pose_pck_accuracy(cln_output.detach().cpu().numpy(), 
                                                            target.detach().cpu().numpy(), 
                                                            mask=target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                                                            thr=pck_thr)
                    cln_per_joint_pck_accs[pck_thr].append(cln_accs)
                    cln_pck_accs[pck_thr].append(cln_avg_acc)
        
                cln_preds, cln_maxvals = keypoints_from_heatmaps(
                    heatmaps=cln_output.detach().cpu().numpy(),
                    center=c, 
                    scale=s,
                    post_process="default", 
                    kernel=11, #  if self.ds_test.heatmap_sigma == 2. else 17, # carefull 
                    target_type="GaussianHeatmap",
                )

                cln_all_preds[idx:idx + num_images, :, 0:2] = cln_preds[:, :, 0:2] # .detach().cpu().numpy()
                cln_all_preds[idx:idx + num_images, :, 2:3] = cln_maxvals # .detach().cpu().numpy()
                cln_all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                cln_all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                cln_all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
                cln_all_boxes[idx:idx + num_images, 5] = score
                cln_all_boxes[idx:idx + num_images, 6] = bbox_id
                cln_image_paths.extend(joints_data['imgPath'])
                cln_mean_loss_test += cln_loss.item()
                cln_mean_acc_test += cln_avg_acc.item()

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
                    adv_accs, adv_avg_acc, adv_cnt = pose_pck_accuracy(adv_output.detach().cpu().numpy(), 
                                                            target.detach().cpu().numpy(), 
                                                            mask=target_weight.detach().cpu().numpy().squeeze(-1) > 0,
                                                            thr=pck_thr)
                    adv_per_joint_pck_accs[pck_thr].append(adv_accs)
                    adv_pck_accs[pck_thr].append(adv_avg_acc)
        
                adv_preds, adv_maxvals = keypoints_from_heatmaps(
                    heatmaps=adv_output.detach().cpu().numpy(),
                    center=c, 
                    scale=s,
                    post_process="default", 
                    kernel=11, #  if self.ds_test.heatmap_sigma == 2. else 17, # carefull 
                    target_type="GaussianHeatmap",
                )

                adv_all_preds[idx:idx + num_images, :, 0:2] = adv_preds[:, :, 0:2] # .detach().cpu().numpy()
                adv_all_preds[idx:idx + num_images, :, 2:3] = adv_maxvals # .detach().cpu().numpy()
                adv_all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                adv_all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                adv_all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
                adv_all_boxes[idx:idx + num_images, 5] = score
                adv_all_boxes[idx:idx + num_images, 6] = bbox_id
                adv_image_paths.extend(joints_data['imgPath'])
                adv_mean_loss_test += adv_loss.item()
                adv_mean_acc_test += adv_avg_acc.item()

                idx += num_images

        cln_mean_loss_test /= self.len_dl_test
        cln_mean_acc_test /= self.len_dl_test
        adv_mean_loss_test /= self.len_dl_test
        adv_mean_acc_test /= self.len_dl_test
        
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
            cln_all_preds[:idx], cln_all_boxes[:idx], cln_image_paths[:idx], res_folder=f'{self.log_path}')
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
            adv_all_preds[:idx], adv_all_boxes[:idx], adv_image_paths[:idx], res_folder=f'{self.log_path}')
        print('Adv AP: ', AP_res)

    def run(self):
        """
        Runs the test.
        """
        # Test

        self._test(1/255., 20, 0.125/255, 'pgd')
        self._test(2/255., 20, 0.25/255, 'pgd')
        self._test(4/255., 20, 0.5/255, 'pgd')
        # self._test(8/255., 20, 1./255, 'pgd')

        print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
