import numpy as np
import torch
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.Train import Train


class COCOTrain(Train):
    """
    COCOTrain class.

    Extension of the Train class for the COCO dataset.
    """

    def __init__(self,
                 exp_name,
                 ds_train,
                 ds_val,
                 epochs=210,
                 batch_size=16,
                 num_workers=4,
                 loss='JointsMSELoss',
                 lr=0.001,
                 lr_decay=True,
                 lr_decay_steps=(170, 200),
                 lr_decay_gamma=0.1,
                 optimizer='Adam',
                 weight_decay=0.,
                 momentum=0.9,
                 nesterov=False,
                 pretrained_weight_path=None,
                 checkpoint_path=None,
                 log_path='./logs',
                 use_tensorboard=True,
                 model_name='poseresnet',
                 model_c=48,
                 model_nof_joints=17,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None
                 ):
        """
        Initializes a new COCOTrain object which extends the parent Train class.
        The initialization function calls the init function of the Train class.

        Args:
            exp_name (str):  experiment name.
            ds_train (HumanPoseEstimationDataset): train dataset.
            ds_val (HumanPoseEstimationDataset): validation dataset.
            epochs (int): number of epochs.
                Default: 210
            batch_size (int): batch size.
                Default: 16
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            lr (float): learning rate.
                Default: 0.001
            lr_decay (bool): learning rate decay.
                Default: True
            lr_decay_steps (tuple): steps for the learning rate decay scheduler.
                Default: (170, 200)
            lr_decay_gamma (float): scale factor for each learning rate decay step.
                Default: 0.1
            optimizer (str): network optimizer. Valid values are 'Adam' and 'SGD'.
                Default: "Adam"
            weight_decay (float): weight decay.
                Default: 0.
            momentum (float): momentum factor.
                Default: 0.9
            nesterov (bool): Nesterov momentum.
                Default: False
            pretrained_weight_path (str): path to pre-trained weights (such as weights from pre-train on imagenet).
                Default: None
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            log_path (str): path where tensorboard data and checkpoints will be saved.
                Default: "./logs"
            use_tensorboard (bool): enables tensorboard use.
                Default: True
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
        super(COCOTrain, self).__init__(
            exp_name=exp_name,
            ds_train=ds_train,
            ds_val=ds_val,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            loss=loss,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_steps=lr_decay_steps,
            lr_decay_gamma=lr_decay_gamma,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            pretrained_weight_path=pretrained_weight_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            use_tensorboard=use_tensorboard,
            model_name=model_name,
            model_c=model_c,
            model_nof_joints=model_nof_joints,
            model_bn_momentum=model_bn_momentum,
            flip_test_images=flip_test_images,
            device=device
        )

    def _train(self):
        running_loss = 0.0
        running_acc = 0.0

        num_samples = len(self.ds_train) # self.len_dl_train * self.batch_size
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        image_paths = []
        idx = 0

        self.model.train()
        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.optim.zero_grad()

            output = self.model(image)

            loss = self.loss_fn(output, target, target_weight)

            loss.backward()

            self.optim.step()

            # Evaluate accuracy
            # Get predictions on the resized images (given as input)
            accs, avg_acc, cnt, joints_preds, joints_target = \
                self.ds_train.evaluate_accuracy(output, target)

            # Original
            num_images = image.shape[0]

            # measure elapsed time
            c = joints_data['center'].numpy()
            s = joints_data['scale'].numpy()
            score = joints_data['score'].numpy()
            pixel_std = 200  # ToDo Parametrize this
            bbox_id = None if not 'bbox_id' in joints_data.keys() else joints_data['bbox_id'].numpy()

            # Get predictions on the original imagee
            preds, maxvals = get_final_preds(True, output.detach(), c, s,
                                             pixel_std)  # ToDo check what post_processing exactly does

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
            all_preds[idx:idx + num_images, :, 2:3] = maxvals.detach().cpu().numpy()
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = bbox_id

            image_paths.extend(joints_data['imgPath'])

            idx += num_images

            print('train_loss', loss.item())
            print('train_acc', avg_acc.item())
            running_acc += avg_acc.item()
            running_loss += loss.item()

        self.loss_train_list.append(running_loss / len(self.dl_train))
        self.acc_train_list.append(running_acc / len(self.dl_train))

        # COCO evaluation
        print('\nTrain AP/AR')
        # all_APs, mAP = self.ds_train.evaluate(
        #     all_preds[:idx], all_boxes[:idx], image_paths[:idx], res_folder=self.log_path)
        all_APs, mAP = self.ds_train.evaluate_overall_accuracy(
            all_preds[:idx], all_boxes[:idx], image_paths[:idx], output_dir=self.log_path)
        
        self.mAP_train_list.append(mAP)
        self.APs_train_list.append(all_APs)

        print(f'Ep{self.epoch} - Train Acc: {self.acc_train_list[-1]:.3f} | Loss: {self.loss_train_list[-1]:.5f} | AP: {self.mAP_train_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('train_loss', self.loss_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_acc', self.acc_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_mAP', self.mAP_train_list[-1],
                                            global_step=self.epoch)
            if self.epoch % 10 == 0: 
                save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
                            self.summary_writer, step=self.epoch, prefix='train_')
                    
    def _val(self):
        running_loss = 0.0
        running_acc = 0.0 

        num_samples = len(self.ds_val)
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        image_paths = []
        idx = 0
        self.model.eval()

        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                output = self.model(image)

                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)

                    output_flipped = flip_back(output_flipped, self.ds_val.flip_pairs)

                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                # Get predictions on the resized images (given as input)
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_val.evaluate_accuracy(output, target)

                # Original
                num_images = image.shape[0]

                # measure elapsed time
                c = joints_data['center'].numpy()
                s = joints_data['scale'].numpy()
                score = joints_data['score'].numpy()
                pixel_std = 200  # ToDo Parametrize this
                bbox_id = None if not 'bbox_id' in joints_data.keys() else joints_data['bbox_id'].numpy()

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

                # print('val_loss', loss.item())
                # print('val_acc', avg_acc.item())
                running_acc += avg_acc.item()
                running_loss += loss.item()

        self.loss_val_list.append(running_loss / len(self.dl_val))
        self.acc_val_list.append(running_acc / len(self.dl_val))

        # COCO evaluation
        print('\nVal AP/AR')
        # all_APs, mAP = self.ds_val.evaluate(
        #     all_preds[:idx], all_boxes[:idx], image_paths[:idx], res_folder=self.log_path)
        all_APs, mAP = self.ds_val.evaluate_overall_accuracy(
            all_preds[:idx], all_boxes[:idx], image_paths[:idx], output_dir=self.log_path)

        self.mAP_val_list.append(mAP)
        self.APs_val_list.append(all_APs)

        print(f'Ep{self.epoch} - Val Acc: {self.acc_val_list[-1]:.3f} | Loss: {self.loss_val_list[-1]:.5f} | AP: {self.mAP_val_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('val_loss', self.loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_acc', self.acc_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_mAP', self.mAP_val_list[-1],
                                            global_step=self.epoch)
            if self.epoch % 10 == 0: 
                save_images(image, target, joints_target, output, joints_preds,
                            joints_data['joints_visibility'], self.summary_writer,
                            step=self.epoch, prefix='val_')
