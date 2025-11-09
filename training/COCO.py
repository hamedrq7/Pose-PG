import numpy as np
import torch
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.Train import Train
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps

class COCO_standard_epoch_info:
    def __init__(self, epoch: int, phase: str, num_samples: int, model_nof_joints: int):
        self.epoch = epoch
        self.phase = phase 

        self.running_loss = 0.0
        self.running_acc = 0.0

        self.all_preds = np.zeros((num_samples, model_nof_joints, 3), dtype=np.float32)
        self.all_boxes = np.zeros((num_samples, 7), dtype=np.float32)
        self.image_paths = []
        self.idx = 0

    def _accumulate_results_for_mAP(self, preds, maxvals, joints_data):
        """
        Store the necessary info for further AP calculation. (for calling DataSet.evaluate()) 

        num_images: int, num samples
        preds: [N, K, 2], the (x,y) location of predicted joints (N=num_samples, K=nof_joints)
        maxvals: [N, K], the scores for predicted joint locations
        joints_data: a dictionary loaded from dataloader
        """
        num_images = preds.shape[0]
        c = joints_data['center'].numpy()
        s = joints_data['scale'].numpy()
        score = joints_data['score'].numpy()
        pixel_std = 200  # ToDo Parametrize this
        bbox_id = None if not 'bbox_id' in joints_data.keys() else joints_data['bbox_id'].numpy()

        # Update accumulated epoch results
        self.all_preds[self.idx:self.idx + num_images, :, 0:2] = preds[:, :, 0:2] # .detach().cpu().numpy()
        self.all_preds[self.idx:self.idx + num_images, :, 2:3] = maxvals # .detach().cpu().numpy()
        self.all_boxes[self.idx:self.idx + num_images, 0:2] = c[:, 0:2]
        self.all_boxes[self.idx:self.idx + num_images, 2:4] = s[:, 0:2]
        self.all_boxes[self.idx:self.idx + num_images, 4] = np.prod(s * pixel_std, 1)
        self.all_boxes[self.idx:self.idx + num_images, 5] = score
        self.all_boxes[self.idx:self.idx + num_images, 6] = bbox_id
        self.image_paths.extend(joints_data['imgPath'])
        self.idx += num_images

    def _accumulate_running_stats(self, loss, accs, avg_acc, cnt):
        """
        accumulate running stats, the MSE loss, and avg pck acc
        """
        self.running_acc += avg_acc.item()
        self.running_loss += loss.item()

    @staticmethod
    def get_pck_acc(output, target, target_weight, pck_thr=0.05):
        """
        ***This is just a helper for shrinking the code volume***
        
        output: [N, K, H, W] predicted heatmaps
        target: [N, K, H, W] groundtruth heatmaps
        target_weights: i think its [N, K] and it indicates visibility and ... for each joint of each image
        pck_thr: treshold for PCK acc
        
        returns the avg pck acc for all joints
        """
        accs, avg_acc, cnt = pose_pck_accuracy(output.detach().cpu().numpy(), 
            target.detach().cpu().numpy(), 
            mask=target_weight.detach().cpu().numpy().squeeze(-1) > 0,
            thr=pck_thr
        )
        return accs, avg_acc, cnt
    
    @staticmethod
    def get_predictions(output_heatmaps, joints_data, post_process="default", kernel=11, target_type="GaussianHeatmap"):
        """
        ***This is just a helper for shrinking the code volume***

        output_heatmaps: [N, K, H, W] predicted heatmaps
        joints_data: the dictionary that dataloader returns (the scale and center is used for getting (x,y) of predictions)

        returns the (x,y) and score of predicted heatmaps (the coordinates are in the original resoultion of image)
        preds: [N, 2], the predicted (x,y) coordinate (in the original image resolution)
        maxvals: [N], score/confidence of each prediction
        """
        preds, maxvals = keypoints_from_heatmaps(
            heatmaps=output_heatmaps.detach().cpu().numpy(),
            center=joints_data['center'].numpy(), 
            scale=joints_data['scale'].numpy(),
            post_process=post_process, 
            kernel=kernel, #  if self.ds_train.heatmap_sigma == 2. else 17, # carefull 
            target_type=target_type,
        )
        return preds, maxvals        
        
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

        epoch_info = COCO_standard_epoch_info(epoch=-1, phase="train", num_samples=len(self.ds_train), model_nof_joints=self.model_nof_joints)

        self.model.train()

        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.optim.zero_grad()
            output = self.model(image) # [N, K, H, W] output heatmaps
            loss = self.loss_fn(output, target, target_weight) # MSE loss
            loss.backward()
            self.optim.step()

            # PCK acc using gt and predicted heatmaps
            accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(output, target, target_weight)
            # Get predictions on the original images
            preds, maxvals = COCO_standard_epoch_info.get_predictions(output, joints_data)
            
            epoch_info._accumulate_results_for_mAP(preds, maxvals, joints_data)
            epoch_info._accumulate_running_stats(loss, accs, avg_acc, cnt)
                
            # print('train_loss', loss.item())
            # print('train_acc', avg_acc.item())

        self.loss_train_list.append(epoch_info.running_loss / len(self.dl_train))
        self.acc_train_list.append(epoch_info.running_acc / len(self.dl_train))

        # COCO evaluation
        print('\nTrain AP/AR')
        all_APs, mAP = self.ds_train.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)

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
            # if self.epoch % 10 == 0: 
            #     save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
            #                 self.summary_writer, step=self.epoch, prefix='train_')
                    
    def _val(self):
        epoch_info = COCO_standard_epoch_info(-1, 'val', len(self.ds_val), self.model_nof_joints)

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
                accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(output, target, target_weight)
                preds, maxvals = COCO_standard_epoch_info.get_predictions(output, joints_data)
                
                epoch_info._accumulate_results_for_mAP(preds, maxvals, joints_data)
                epoch_info._accumulate_running_stats(loss, accs, avg_acc, cnt)

                # print('val_loss', loss.item())
                # print('val_acc', avg_acc.item())

        self.loss_val_list.append(epoch_info.running_loss / len(self.dl_val))
        self.acc_val_list.append(epoch_info.running_acc / len(self.dl_val))

        # COCO evaluation
        print('\nVal AP/AR')
        all_APs, mAP = self.ds_val.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)
       
       
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
            # if self.epoch % 10 == 0: 
            #     save_images(image, target, joints_target, output, joints_preds,
            #                 joints_data['joints_visibility'], self.summary_writer,
            #                 step=self.epoch, prefix='val_')
