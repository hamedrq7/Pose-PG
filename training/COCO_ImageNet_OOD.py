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

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

class COCO_ImageNet_OOD(TrainAuxilary):
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
        super(COCO_ImageNet_OOD, self).__init__(
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

        self.train_aux_iterator = inf_generator(self.aux_dl_train)
        self.val_aux_iterator = inf_generator(self.aux_dl_val)

        self.ood_acc_train_list = [] 
        self.ood_acc_val_list = []

    def _create_aux_dataloaders(self): 
        aux_dl_train = DataLoader(self.aux_ds_train, batch_size=self.aux_batch_size, shuffle=True,
                                   num_workers=self.num_workers, drop_last=False, pin_memory=True)
        
        aux_len_dl_train = len(aux_dl_train)
        aux_dl_val = DataLoader(self.aux_ds_val, batch_size=self.aux_batch_size, shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        aux_len_dl_val = len(aux_dl_val)

        return aux_dl_train, aux_len_dl_train, aux_dl_val, aux_len_dl_val
    
    def _get_results_dict(self): 
        base = self._get_base_results_dict()
        base['ood_acc_train_list'] = self.ood_acc_train_list
        base['ood_acc_val_list'] = self.ood_acc_val_list
        return base 
    
    def _train(self):

        epoch_info = COCO_standard_epoch_info(epoch=-1, phase="train", num_samples=len(self.pose_ds_train), model_nof_joints=self.model_nof_joints)

        self.model.train()

        ood_running_loss = 0.0
        ood_running_corrects = 0 
        total_running_loss = 0.0 
        num_ood_samples = 0 

        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.pose_dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.optim.zero_grad()
            
            # First pass the coco samples
            coco_ood_output, coco_pose_output = self.model(image) # feats: [N, C, H, W], output: [N, K, H, W] output heatmaps
            pose_loss = self.pose_loss_fn(coco_pose_output, target, target_weight) # MSE loss
            
            # Second pass imagenet samples 
            # but first you need to load imagenet
            imagenet_images, _ = self.train_aux_iterator.__next__()
            imagenet_images = imagenet_images.to(self.device)

            imagenet_ood_output, _ = self.model(imagenet_images)

            # Now you need to create binary labels for OOD and IID samples. 
            # IID=1, OOD=0 
            domain_id   = torch.ones(image.shape[0], dtype=torch.float32, device=self.device)
            domain_ood  = torch.zeros(imagenet_images.shape[0], dtype=torch.float32, device=self.device)

            ood_out = torch.cat([coco_ood_output, imagenet_ood_output], dim=0)
            ood_label = torch.cat([domain_id, domain_ood], dim=0)
            ood_loss = self.aux_loss_fn(ood_out, ood_label)

            loss = pose_loss + self.aux_loss_weight * ood_loss

            loss.backward()
            self.optim.step()

            # PCK acc using gt and predicted heatmaps
            accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(coco_pose_output, target, target_weight)
            # Get predictions on the original images
            preds, maxvals = COCO_standard_epoch_info.get_predictions(coco_pose_output, joints_data)
            
            epoch_info._accumulate_results_for_mAP(preds, maxvals, joints_data)
            epoch_info._accumulate_running_stats(loss, accs, avg_acc, cnt)
            
            # Storing OOD stuff: 
            prob = torch.sigmoid(ood_out)
            pred = (prob >= 0.5).float()
            ood_running_corrects += (pred == ood_label).sum().item()
            ood_running_loss += ood_loss.item()
            total_running_loss += loss.item()
            num_ood_samples += ood_label.shape[0]

        self.pose_loss_train_list.append(epoch_info.running_loss / len(self.pose_dl_train))
        self.pose_acc_train_list.append(epoch_info.running_acc / len(self.pose_dl_train))
        self.aux_loss_train_list.append(ood_running_loss / len(self.pose_dl_train))
        self.total_loss_train_list.append(total_running_loss / len(self.pose_dl_train))
        self.ood_acc_train_list.append(ood_running_corrects / num_ood_samples)

        # COCO evaluation
        print('\nTrain AP/AR')
        all_APs, mAP = self.ds_train.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)

        self.pose_mAP_train_list.append(mAP)
        self.pose_APs_train_list.append(all_APs)

        print(f'Ep{self.epoch} - Train Pose Acc: {self.pose_acc_train_list[-1]:.3f} | Pose Loss: {self.pose_loss_train_list[-1]:.5f} | Pose AP: {self.pose_mAP_train_list[-1]:.3f}')
        print(f'Train OOD Acc: {self.ood_acc_train_list[-1]:.3f} | OOD Loss: {self.aux_loss_train_list[-1]:.5f} | Total Loss: {self.total_loss_train_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('train_loss', self.pose_loss_train_list[-1], # I regard this as Pose 
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_acc', self.pose_acc_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_mAP', self.pose_mAP_train_list[-1],
                                            global_step=self.epoch)
            
            self.summary_writer.add_scalar('train_total_loss', self.total_loss_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_ood_loss', self.aux_loss_train_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('train_ood_acc', self.ood_acc_train_list[-1],
                                            global_step=self.epoch)

    def _val(self):
        epoch_info = COCO_standard_epoch_info(-1, 'val', len(self.ds_val), self.model_nof_joints)

        self.model.eval()

        all_coco_ood_outputs = []
        all_imagenet_ood_outputs = []
        # For evaluation, first pass all samples of COCO, then pass all val samples of ImageNet
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.pose_dl_val, desc='COCO Validating')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                coco_ood_output, coco_pose_output = self.model(image)

                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)
                    output_flipped = flip_back(output_flipped, self.ds_val.flip_pairs)
                    coco_pose_output = (coco_pose_output + output_flipped) * 0.5

                poes_loss = self.pose_loss_fn(coco_pose_output, target, target_weight)

                # Evaluate accuracy
                accs, avg_acc, cnt = COCO_standard_epoch_info.get_pck_acc(coco_pose_output, target, target_weight)
                preds, maxvals = COCO_standard_epoch_info.get_predictions(coco_pose_output, joints_data)
                
                epoch_info._accumulate_results_for_mAP(preds, maxvals, joints_data)
                epoch_info._accumulate_running_stats(poes_loss, accs, avg_acc, cnt)

                all_coco_ood_outputs.append(coco_ood_output.cpu().detach())
            
            for step, (image, _) in enumerate(tqdm(self.aux_dl_val, desc='ImageNet Validating')): 
                image = image.to(self.device)
                imagenet_ood_output, _ = self.model(image)
                all_imagenet_ood_outputs.append(imagenet_ood_output.cpu().detach())
            
            # Process ood outputs
            all_coco_ood_outputs = torch.tensor(all_coco_ood_outputs)
            all_imagenet_ood_outputs = torch.tensor(all_imagenet_ood_outputs)
            
            domain_id   = torch.ones(all_coco_ood_outputs.shape[0], dtype=torch.float32, device=self.device)
            domain_ood  = torch.zeros(all_imagenet_ood_outputs.shape[0], dtype=torch.float32, device=self.device)

            ood_out = torch.cat([all_coco_ood_outputs, all_imagenet_ood_outputs], dim=0)
            ood_label = torch.cat([domain_id, domain_ood], dim=0)
            ood_loss = self.aux_loss_fn(ood_out, ood_label).item()
        
            prob = torch.sigmoid(ood_out)
            pred = (prob >= 0.5).float()
            ood_corrects = (pred == ood_label).sum().item()
            num_ood_samples = ood_label.shape[0]

        self.aux_loss_val_list.append(ood_loss)
        self.ood_acc_val_list.append(ood_corrects / num_ood_samples)
        self.pose_loss_val_list.append(epoch_info.running_loss / len(self.pose_dl_val))
        self.pose_acc_val_list.append(epoch_info.running_acc / len(self.pose_dl_val))
        self.total_loss_val_list.append((epoch_info.running_loss / len(self.pose_dl_val)) + ood_loss*self.aux_loss_weight)

        # COCO evaluation
        print('\nVal AP/AR')
        all_APs, mAP = self.ds_val.evaluate(
            epoch_info.all_preds[:epoch_info.idx], epoch_info.all_boxes[:epoch_info.idx], epoch_info.image_paths[:epoch_info.idx], res_folder=self.log_path)
       
        self.pose_mAP_val_list.append(mAP)
        self.pose_APs_val_list.append(all_APs)

        print(f'Ep{self.epoch} - Val Acc: {self.pose_acc_val_list[-1]:.3f} | Loss: {self.pose_loss_val_list[-1]:.5f} | AP: {self.pose_mAP_val_list[-1]:.3f}')
        print(f'Val OOD Acc: {self.ood_acc_val_list[-1]:.3f} | OOD Loss: {self.aux_loss_val_list[-1]:.5f} | Total Loss: {self.total_loss_val_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('val_loss', self.pose_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_acc', self.pose_acc_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_mAP', self.pose_mAP_val_list[-1],
                                            global_step=self.epoch)
            
            self.summary_writer.add_scalar('val_total_loss', self.total_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_ood_loss', self.aux_loss_val_list[-1],
                                            global_step=self.epoch)
            self.summary_writer.add_scalar('val_ood_acc', self.ood_acc_val_list[-1],
                                            global_step=self.epoch)