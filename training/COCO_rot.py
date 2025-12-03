import numpy as np
import torch
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.TrainAuxilary import TrainAuxilary
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps
from training.COCO import COCO_standard_epoch_info
from torch.utils.data.dataloader import DataLoader
import sklearn
from sklearn.metrics import confusion_matrix

class COCO_rot(TrainAuxilary):
    """
    COCOTrain class.

    Extension of the Train class for the COCO dataset.
    """

    def __init__(self,
                 exp_name,
                 pose_ds_train,
                 pose_ds_val,
                 aux_ds_train,
                 aux_ds_val,
                 aux_loss_weight: float, 
                 aux_loss: str = 'BCE', 
                 epochs=210,
                 pose_batch_size=16,
                 aux_batch_size=16,
                 num_workers=4,
                 pose_loss='JointsMSELoss',
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
                 model_name='poseresnet_ood',
                 model_c=48,
                 model_nof_joints=17,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None
                 ):
        """
        Add
        """
        super(COCO_rot, self).__init__(
            exp_name=exp_name,
            pose_ds_train = pose_ds_train,
            pose_ds_val = pose_ds_val,
            aux_ds_train = aux_ds_train,
            aux_ds_val = aux_ds_val,
            aux_loss = aux_loss,
            aux_loss_weight = aux_loss_weight,
            epochs=epochs,
            pose_batch_size=pose_batch_size,
            aux_batch_size=aux_batch_size,
            num_workers=num_workers,
            pose_loss=pose_loss,
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

        self.rot_acc_train_list = [] 
        self.rot_acc_val_list = []

    def _create_aux_dataloaders(self): 
        # No use
        return None, None, None, None
    
    def _get_results_dict(self): 
        base = self._get_base_results_dict()
        base['rot_acc_train_list'] = self.rot_acc_train_list
        base['rot_acc_val_list'] = self.rot_acc_val_list
        return base 
    
    def _train(self):

        epoch_info = COCO_standard_epoch_info(epoch=-1, phase="train", num_samples=len(self.pose_ds_train), model_nof_joints=self.model_nof_joints)

        self.model.train()

        rot_running_loss = 0.0
        rot_running_corrects = 0 
        total_running_loss = 0.0 
        num_rot_samples = 0 

        for step, (batch_hpe, batch_rot_0, batch_rot_180) in enumerate(tqdm(self.pose_dl_train, desc='Training')):
            pose_image, pose_target, pose_target_weight, pose_joints_data = batch_hpe
            pose_image = pose_image.to(self.device)
            pose_target = pose_target.to(self.device)
            pose_target_weight = pose_target_weight.to(self.device)

            self.optim.zero_grad()
            
            # The HPE branch: 
            _, pose_output = self.model(pose_image)
            pose_loss = self.pose_loss_fn(pose_output, pose_target, pose_target_weight) # MSE loss
            
            # Second Branch for Rotation Prediction
            x0, _, _, _ = batch_rot_0
            x0 = x0.to(self.device)

            x180, _, _, _ = batch_rot_180
            x180 = x180.to(self.device)
            
            x_rot = torch.cat([x0, x180], dim=0)

            rot_output, _ = self.model(x_rot)

            # Now you need to create binary labels for rotation prediction. 
            # rot0=1, rot180=0 
            label_rot0   = torch.ones(x0.shape[0], dtype=torch.float32, device=self.device)
            label_rot180  = torch.zeros(x180.shape[0], dtype=torch.float32, device=self.device)
            rot_label = torch.cat([label_rot0, label_rot180], dim=0)

            rot_loss = self.aux_loss_fn(rot_output, rot_label)

            if self.epoch >= 0: 
                loss = pose_loss + self.aux_loss_weight * rot_loss
            else: 
                loss = pose_loss

            loss.backward()
            self.optim.step()

            # PCK acc using gt and predicted heatmaps
            accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(pose_output, pose_target, pose_target_weight)
            # Get predictions on the original images
            preds, maxvals = COCO_standard_epoch_info.get_predictions(pose_output, pose_joints_data)
            
            epoch_info._accumulate_results_for_mAP(preds, maxvals, pose_joints_data)
            epoch_info._accumulate_running_stats(pose_loss, accs, avg_acc, cnt)
            
            # Storing OOD stuff: 
            prob = torch.sigmoid(rot_output)
            pred = (prob >= 0.5).float()
            rot_running_corrects += (pred == rot_label).sum().item()
            rot_running_loss += rot_loss.item()
            total_running_loss += loss.item()
            num_rot_samples += rot_label.shape[0]

            # print('Pose acc', avg_acc)
            # print('Pose Loss', pose_loss.item())
            # print('OOD acc', (pred == ood_label).sum().item()/ood_label.shape[0])
            # print('OOD Loss', ood_loss.item())
            
            # if step > 1: 
            #     break 

        self.pose_loss_train_list.append(epoch_info.running_loss / len(self.pose_dl_train))
        self.pose_acc_train_list.append(epoch_info.running_acc / len(self.pose_dl_train))
        self.aux_loss_train_list.append(rot_running_loss / len(self.pose_dl_train))
        self.total_loss_train_list.append(total_running_loss / len(self.pose_dl_train))
        self.ood_acc_train_list.append(rot_running_corrects / num_rot_samples)

        # COCO evaluation
        print('\nTrain AP/AR')
        all_APs, mAP = self.pose_ds_train.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)

        self.pose_mAP_train_list.append(mAP)
        self.pose_APs_train_list.append(all_APs)

        print(f'Ep{self.epoch} - Train Pose Acc: {self.pose_acc_train_list[-1]:.3f} | Pose Loss: {self.pose_loss_train_list[-1]:.5f} | Pose AP: {self.pose_mAP_train_list[-1]:.3f}')
        print(f'Train ROT Acc: {self.ood_acc_train_list[-1]:.3f} | ROT Loss: {self.aux_loss_train_list[-1]:.5f} | Total Loss: {self.total_loss_train_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('train_loss', self.pose_loss_train_list[-1], # I regard this as Pose 
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_acc', self.pose_acc_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_mAP', self.pose_mAP_train_list[-1],
                                            global_step=self.epoch)
            
            self.summary_writer.add_scalar('train_total_loss', self.total_loss_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_rot_loss', self.aux_loss_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_rot_acc', self.ood_acc_train_list[-1],
                                            global_step=self.epoch)

    def _print_conf_mat(self, pred_class, ood_label): 
        pred_np = pred_class.cpu().numpy()
        label_np = ood_label.cpu().numpy()
        
        cm = confusion_matrix(label_np, pred_np)
        print(cm)

    def _val(self):
        epoch_info = COCO_standard_epoch_info(-1, 'val', len(self.pose_ds_val), self.model_nof_joints)

        self.model.eval()

        rot_running_loss = 0.0
        rot_running_corrects = 0 
        total_running_loss = 0.0 
        num_rot_samples = 0 

        # For evaluation, first pass all samples of COCO, then pass all val samples of ImageNet
        with torch.no_grad():
            for step, (batch_hpe, batch_rot_0, batch_rot_180) in enumerate(tqdm(self.pose_dl_val, desc='COCO Validating')):
                pose_image, pose_target, pose_target_weight, pose_joints_data = batch_hpe
                # pose_image = pose_image.to(self.device)
                # pose_target = pose_target.to(self.device)
                # pose_target_weight = pose_target_weight.to(self.device)

                _, pose_output = self.model(pose_image)

                if self.flip_test_images:
                    image_flipped = flip_tensor(pose_image, dim=-1)
                    _, output_flipped = self.model(image_flipped)
                    output_flipped = flip_back(output_flipped, self.pose_ds_val.flip_pairs)
                    pose_output = (pose_output + output_flipped) * 0.5

                pose_loss = self.pose_loss_fn(pose_output, pose_target, pose_target_weight)

                # Evaluate accuracy
                accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(pose_output, pose_target, pose_target_weight)
                preds, maxvals = COCO_standard_epoch_info.get_predictions(pose_output, pose_joints_data)
                
                epoch_info._accumulate_results_for_mAP(preds, maxvals, pose_joints_data)
                epoch_info._accumulate_running_stats(pose_loss, accs, avg_acc, cnt)

                # Second Branch for Rotation Prediction
                x0, _, _, _ = batch_rot_0
                x0 = x0.to(self.device)

                x180, _, _, _ = batch_rot_180
                x180 = x180.to(self.device)
                
                x_rot = torch.cat([x0, x180], dim=0)

                rot_output, _ = self.model(x_rot)
                label_rot0   = torch.ones(x0.shape[0], dtype=torch.float32, device=self.device)
                label_rot180  = torch.zeros(x180.shape[0], dtype=torch.float32, device=self.device)
                rot_label = torch.cat([label_rot0, label_rot180], dim=0)

                rot_loss = self.aux_loss_fn(rot_output, rot_label)
                
                prob = torch.sigmoid(rot_output)
                pred = (prob >= 0.5).float()
                rot_running_corrects += (pred == rot_label).sum().item()
                rot_running_loss += rot_loss.item()
                total_running_loss += (pose_loss + self.aux_loss_weight * rot_loss).item()
                num_rot_samples += rot_label.shape[0]


                self._print_conf_mat(pred, rot_label)

        self.aux_loss_val_list.append(rot_running_loss / len(self.pose_dl_val))
        self.ood_acc_val_list.append(rot_running_corrects / num_rot_samples)
        self.pose_loss_val_list.append(epoch_info.running_loss / len(self.pose_dl_val))
        self.pose_acc_val_list.append(epoch_info.running_acc / len(self.pose_dl_val))
        self.total_loss_val_list.append(total_running_loss / len(self.pose_dl_val))

        # COCO evaluation
        print('\nVal AP/AR')
        all_APs, mAP = self.pose_ds_val.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)
       
        self.pose_mAP_val_list.append(mAP)
        self.pose_APs_val_list.append(all_APs)

        print(f'Ep{self.epoch} - Val Acc: {self.pose_acc_val_list[-1]:.3f} | Loss: {self.pose_loss_val_list[-1]:.5f} | AP: {self.pose_mAP_val_list[-1]:.3f}')
        print(f'Val ROT Acc: {self.ood_acc_val_list[-1]:.3f} | ROT Loss: {self.aux_loss_val_list[-1]:.5f} | Total Loss: {self.total_loss_val_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('val_loss', self.pose_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_acc', self.pose_acc_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_mAP', self.pose_mAP_val_list[-1],
                                            global_step=self.epoch)
            
            self.summary_writer.add_scalar('val_total_loss', self.total_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_rot_loss', self.aux_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_rot_acc', self.ood_acc_val_list[-1],
                                            global_step=self.epoch)