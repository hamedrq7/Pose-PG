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
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet

import numpy as np 

class Test(object):
    """
    Test class.

    The class provides a basic tool for testing HRNet checkpoints.

    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 ds_test,
                 batch_size=1,
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

        # torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

        print(self.device)

        #
        # load model
        if model_name == 'hrnet':
            self.model = HRNet(c=self.model_c, nof_joints=self.model_nof_joints,
                            bn_momentum=self.model_bn_momentum).to(self.device)
        elif model_name == 'poseresnet':
            self.model = PoseResNet(resnet_size=self.model_c, nof_joints=self.model_nof_joints, bn_momentum=self.model_bn_momentum).to(self.device)
        else:
            print('invalid model name')

        # define loss
        if self.loss == 'JointsMSELoss':
            self.loss_fn = JointsMSELoss().to(self.device)
        elif self.loss == 'JointsOHKMMSELoss':
            self.loss_fn = JointsOHKMMSELoss().to(self.device)
        else:
            raise NotImplementedError

        if not pre_trained_only: 
            # load previous checkpoint
            if self.checkpoint_path is not None:
                print('Loading checkpoint %s...' % self.checkpoint_path)
                if os.path.isdir(self.checkpoint_path):
                    path = os.path.join(self.checkpoint_path, 'checkpoint_last.pth')
                else:
                    path = self.checkpoint_path
                self.starting_epoch, self.model, _, self.params = load_checkpoint(path, self.model, device=self.device)
            else:
                raise ValueError('checkpoint_path is not defined')
        else: 
            # load pre-trained weights (such as those pre-trained on imagenet)
            if pretrained_weight_path is not None:
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    torch.load(pretrained_weight_path, map_location=self.device),
                    strict=False  # strict=False is required to load models pre-trained on imagenet
                )
                print('Pre-trained weights loaded.')
                if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                    print('Pre-trained weights missing keys:', missing_keys)
                    print('Pre-trained weights unexpected keys:', unexpected_keys)

                self.starting_epoch = 0
                self.params = None
        #
        # load test dataset
        self.dl_test = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_test = len(self.dl_test)

        #
        # initialize variables
        self.mean_loss_test = 0.
        self.mean_acc_test = 0.

    def _test(self):
        num_samples = len(self.dl_test)
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
                print('output.shape', output.shape)
                print('target.shape', target.shape)
                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)

                    output_flipped = flip_back(output_flipped, self.ds_test.flip_pairs)

                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                # Get predictions on the input
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_test.evaluate_accuracy(output, target)

                num_images = image.shape[0]

                # measure elapsed time
                c = joints_data['center'].numpy()
                s = joints_data['scale'].numpy()
                score = joints_data['score'].numpy()
                pixel_std = 200  # ToDo Parametrize this
                bbox_id = joints_data['bbox_id'].numpy()

                preds, maxvals = get_final_preds(True, output, c, s,
                                                 pixel_std)  # ToDo check what post_processing exactly does

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
                all_preds[idx:idx + num_images, :, 2:3] = maxvals.detach().cpu().numpy()
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

                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'])

                if step >= 3:
                    break

        self.mean_loss_test /= self.len_dl_test
        self.mean_acc_test /= self.len_dl_test

        print('\nTest: Loss %f - Accuracy %f' % (self.mean_loss_test, self.mean_acc_test))
        print('\nVal AP/AR')
        print(all_preds[:idx], all_boxes[:idx])
        val_acc, mean_mAP_val = self.ds_test.evaluate( # evaluate_overall_accuracy
            all_preds[:idx], all_boxes[:idx], image_paths[:idx], res_folder='/')
        print('val_acc', val_acc, 'mean_mAP_val', mean_mAP_val)

    # def _test(self):
    #     num_samples = len(self.dl_test)
    #     all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
    #     all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
    #     image_paths = []
    #     idx = 0
    #     self.model.eval()

    #     with torch.no_grad():
    #         for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_test)):
    #             image = image.to(self.device)
    #             target = target.to(self.device)
    #             target_weight = target_weight.to(self.device)

    #             output = self.model(image)
    #             # print('output.shape', output.shape)
    #             # print('target.shape', target.shape)
    #             if self.flip_test_images:
    #                 image_flipped = flip_tensor(image, dim=-1)
    #                 output_flipped = self.model(image_flipped)

    #                 output_flipped = flip_back(output_flipped, self.ds_test.flip_pairs)

    #                 output = (output + output_flipped) * 0.5

    #             loss = self.loss_fn(output, target, target_weight)

    #             # Evaluate accuracy
    #             # Get predictions on the input
    #             accs, avg_acc, cnt, joints_preds, joints_target = \
    #                 self.ds_test.evaluate_accuracy(output, target)

    #             num_images = image.shape[0]

    #             # measure elapsed time
    #             c = joints_data['center'].numpy()
    #             s = joints_data['scale'].numpy()
    #             score = joints_data['score'].numpy()
    #             pixel_std = 200  # ToDo Parametrize this

    #             preds, maxvals = get_final_preds(True, output, c, s,
    #                                              pixel_std)  # ToDo check what post_processing exactly does

    #             all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
    #             all_preds[idx:idx + num_images, :, 2:3] = maxvals.detach().cpu().numpy()
    #             # double check this all_boxes parts
    #             all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
    #             all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
    #             all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
    #             all_boxes[idx:idx + num_images, 5] = score
    #             image_paths.extend(joints_data['imgPath'])

    #             idx += num_images

    #             self.mean_loss_test += loss.item()
    #             self.mean_acc_test += avg_acc.item()

    #             if step == 0:
    #                 save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'])



    #     self.mean_loss_test /= self.len_dl_test
    #     self.mean_acc_test /= self.len_dl_test

    #     print('\nTest: Loss %f - Accuracy %f' % (self.mean_loss_test, self.mean_acc_test))
    #     print('\nVal AP/AR')
        
    #     val_acc, mean_mAP_val = self.ds_test.evaluate(
    #         all_preds, all_boxes, image_paths, res_folder='/')
    #     print('val_acc', val_acc, 'mean_mAP_val', mean_mAP_val)

    def run(self):
        """
        Runs the test.
        """

        print('\nTest started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # start testing
        print('\nLoaded checkpoint %s @ %s\nSaved epoch %d' %
              (self.checkpoint_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.starting_epoch))

        self.mean_loss_test = 0.
        self.mean_acc_test = 0.

        #
        # Test

        self._test()

        print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
