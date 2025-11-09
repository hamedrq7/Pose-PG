import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.Train import Train
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps
from training.COCO import COCO_standard_epoch_info     
from misc.general_utils import NormalizeByChannelMeanStd
from misc.context import ctx_noparamgrad_and_eval 
from misc.general_utils import perturb

class COCOAdv_Train(Train):
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
                 device=None,
                 mix_st: bool = False, 
                 attack_type: str = 'pgd',
                 epsilon: float = 8/255.,
                 num_steps: int = 10,
                 step_size: float = 2/255.,
                 rand_init: bool = True,
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225],
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
        super(COCOAdv_Train, self).__init__(
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
            device=device,
        )
        self.mix_st = mix_st
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand_init = rand_init
        self.mean = mean
        self.std = std

        # Model is already instantiated in the Train class init function, 
        # Here you cann add the normalization layer
        # be carefull this will change the state_dict when saving the model
        # Also, since the normalization layer you add here has no learnable parameters, its fine
        # But, if you add any learnable parameters here to the model, the optimizers wouldnt know about it,
        # So be carefull. 
        self.model = nn.Sequential(
            NormalizeByChannelMeanStd(self.mean, self.std).to(self.device),
            self.model
        ).to(self.device)


        # For monitoring adv and clean stats
        # Note the parent class, Train, has some lists for monitoring but 
        # BUT only the self.loss_train_list and self.loss_val_list is used in this class
        # And they amount to the total loss
        self.cln_loss_train_list = []
        self.cln_loss_val_list = [] 
        self.cln_acc_train_list = [] 
        self.cln_acc_val_list = [] 
        self.cln_mAP_train_list = [] 
        self.cln_mAP_val_list = []
        self.cln_APs_train_list = []
        self.cln_APs_val_list = []

        self.adv_loss_train_list = [] 
        self.adv_loss_val_list = [] 
        self.adv_acc_train_list = [] 
        self.adv_acc_val_list = [] 
        self.adv_mAP_train_list = []  
        self.adv_mAP_val_list = [] 
        self.adv_APs_train_list = []  
        self.adv_APs_val_list = [] 

    def _train(self):
        running_loss_total = 0.0 
        
        cln_epoch_info = COCO_standard_epoch_info(epoch=-1, phase="cln_train", num_samples=len(self.ds_train), model_nof_joints=self.model_nof_joints)
        adv_epoch_info = COCO_standard_epoch_info(epoch=-1, phase="adv_train", num_samples=len(self.ds_train), model_nof_joints=self.model_nof_joints)

        self.model.train()

        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            # Generate adversarial samples 
            with ctx_noparamgrad_and_eval(self.model):
                adv_image = perturb(self.model, self.device, image, target, target_weight,
                                    loss_fn = self.loss_fn, epsilon=self.epsilon, num_steps=self.num_steps, 
                                    step_size=self.step_size, rand_init=self.rand_init)

            self.optim.zero_grad()

            cln_output = self.model(image) # [N, K, H, W] 
            adv_output = self.model(adv_image)
            
            cln_loss = self.loss_fn(cln_output, target, target_weight) # MSE loss
            adv_loss = self.loss_fn(adv_output, target, target_weight)            
            loss = (cln_loss+adv_loss)/2.0

            loss.backward()
            self.optim.step()

            # PCK acc using gt and predicted heatmaps
            cln_accs, cln_avg_acc, cln_cnt = COCO_standard_epoch_info.get_pck_acc(cln_output, target, target_weight)
            adv_accs, adv_avg_acc, adv_cnt = COCO_standard_epoch_info.get_pck_acc(adv_output, target, target_weight)
            
            # Get predictions on the original images
            cln_preds, cln_maxvals = COCO_standard_epoch_info.get_predictions(cln_output, joints_data)
            adv_preds, adv_maxvals = COCO_standard_epoch_info.get_predictions(adv_output, joints_data)
            
            cln_epoch_info._accumulate_results_for_mAP(cln_preds, cln_maxvals, joints_data)
            adv_epoch_info._accumulate_results_for_mAP(adv_preds, adv_maxvals, joints_data)

            cln_epoch_info._accumulate_running_stats(cln_loss, cln_accs, cln_avg_acc, cln_cnt)
            adv_epoch_info._accumulate_running_stats(adv_loss, adv_accs, adv_avg_acc, adv_cnt)
            
            running_loss_total += loss.item()
            print(f'Total Loss: {loss.item()[-1]:.5f}')
            print(f'CLN Loss: {cln_loss.item()[-1]:.5f}')
            print(f'ADV Loss: {adv_loss.item()[-1]:.5f}')
            print(f'CLN Acc: {cln_avg_acc.item()[-1]:.5f}')
            print(f'Adv Acc: {adv_avg_acc.item()[-1]:.5f}')
            print('')
            
        self.loss_train_list.append(running_loss_total / len(self.dl_train))
        self.cln_loss_train_list.append(cln_epoch_info.running_loss / len(self.dl_train))
        self.adv_loss_train_list.append(adv_epoch_info.running_loss / len(self.dl_train))
        self.cln_acc_train_list.append(cln_epoch_info.running_acc / len(self.dl_train))
        self.adv_acc_train_list.append(adv_epoch_info.running_acc / len(self.dl_train))
        
        # COCO evaluation
        print('\nTrain CLEAN AP/AR')
        cln_all_APs, cln_mAP = self.ds_train.evaluate(
            cln_epoch_info.all_preds[:cln_epoch_info.idx], cln_epoch_info.all_boxes[:cln_epoch_info.idx], cln_epoch_info.image_paths[:cln_epoch_info.idx], res_folder=self.log_path, 
            filename_prefix='cln_train')

        self.cln_mAP_train_list.append(cln_mAP)
        self.cln_APs_train_list.append(cln_all_APs)

        print('\nTrain ADV AP/AR')
        adv_all_APs, adv_mAP = self.ds_train.evaluate(
            adv_epoch_info.all_preds[:adv_epoch_info.idx], adv_epoch_info.all_boxes[:adv_epoch_info.idx], adv_epoch_info.image_paths[:adv_epoch_info.idx], res_folder=self.log_path, 
            filename_prefix='adv_train')

        self.adv_mAP_train_list.append(adv_mAP)
        self.adv_APs_train_list.append(adv_all_APs)

        
        print(f'Ep{self.epoch} - Clean Train Acc: {self.cln_acc_train_list[-1]:.3f} | Loss: {self.cln_loss_train_list[-1]:.5f} | AP: {self.cln_mAP_train_list[-1]:.3f}')
        print(f'                 Adv   Train Acc: {self.adv_acc_train_list[-1]:.3f} | Loss: {self.adv_loss_train_list[-1]:.5f} | AP: {self.adv_mAP_train_list[-1]:.3f}')
        print(f'                 Total Loss: {self.loss_train_list[-1]:.3f}')
        
        if self.use_tensorboard:
            self.summary_writer.add_scalar('train_loss', self.cln_loss_train_list[-1],
                                            global_step=self.epoch) # This is for clean
            self.summary_writer.add_scalar('train_acc', self.cln_acc_train_list[-1],
                                            global_step=self.epoch) # This is for clean
            self.summary_writer.add_scalar('train_mAP', self.cln_mAP_train_list[-1],
                                            global_step=self.epoch) # This is for clean
            
            self.summary_writer.add_scalar('Adv_train_loss', self.adv_loss_train_list[-1],
                                            global_step=self.epoch) 
            self.summary_writer.add_scalar('Adv_train_acc', self.adv_acc_train_list[-1],
                                            global_step=self.epoch) 
            self.summary_writer.add_scalar('Adv_train_mAP', self.adv_mAP_train_list[-1],
                                            global_step=self.epoch) 
            
            self.summary_writer.add_scalar('total_train_loss', self.loss_train_list[-1],
                                            global_step=self.epoch)
            # if self.epoch % 10 == 0: 
            #     save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
            #                 self.summary_writer, step=self.epoch, prefix='train_')
                    
    def _val(self):
        running_loss_total = 0.0
        cln_epoch_info = COCO_standard_epoch_info(-1, 'cln_val', len(self.ds_val), self.model_nof_joints)
        adv_epoch_info = COCO_standard_epoch_info(-1, 'adv_val', len(self.ds_val), self.model_nof_joints)

        self.model.eval()

        
        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)
            
            # get adv inputs: 
            with ctx_noparamgrad_and_eval(self.model): 
                adv_image = perturb(self.model, self.device, image, target, target_weight,
                                    loss_fn = self.loss_fn, epsilon=self.epsilon, num_steps=self.num_steps, 
                                    step_size=self.step_size, rand_init=self.rand_init)

            with torch.no_grad():
                cln_output = self.model(image)
                adv_output = self.model(adv_image)

                if self.flip_test_images:
                    cln_image_flipped = flip_tensor(image, dim=-1)
                    cln_output_flipped = self.model(cln_image_flipped)
                    cln_output_flipped = flip_back(cln_output_flipped, self.ds_val.flip_pairs)
                    cln_output = (cln_output + cln_output_flipped) * 0.5
                    
                    adv_image_flipped = flip_tensor(adv_image, dim=-1)
                    adv_output_flipped = self.model(adv_image_flipped)
                    adv_output_flipped = flip_back(adv_output_flipped, self.ds_val.flip_pairs)
                    adv_output = (adv_output + adv_output_flipped) * 0.5
                    
                cln_loss = self.loss_fn(cln_output, target, target_weight)
                adv_loss = self.loss_fn(adv_output, target, target_weight)  
                loss = (cln_loss+adv_loss)/2.0

            # Evaluate accuracy
            cln_accs, cln_avg_acc, cln_cnt = COCO_standard_epoch_info.get_pck_acc(cln_output, target, target_weight)
            adv_accs, adv_avg_acc, adv_cnt = COCO_standard_epoch_info.get_pck_acc(adv_output, target, target_weight)
            cln_preds, cln_maxvals = COCO_standard_epoch_info.get_predictions(cln_output, joints_data)
            adv_preds, adv_maxvals = COCO_standard_epoch_info.get_predictions(adv_output, joints_data)
            
            cln_epoch_info._accumulate_results_for_mAP(cln_preds, cln_maxvals, joints_data)
            adv_epoch_info._accumulate_results_for_mAP(adv_preds, adv_maxvals, joints_data)
            cln_epoch_info._accumulate_running_stats(cln_loss, cln_accs, cln_avg_acc, cln_cnt)
            adv_epoch_info._accumulate_running_stats(adv_loss, adv_accs, adv_avg_acc, adv_cnt)

            running_loss_total += loss.item()
            # print('val_loss', loss.item())
            # print('val_acc', avg_acc.item())

        self.loss_val_list.append(running_loss_total / len(self.dl_val))
        self.cln_loss_val_list.append(cln_epoch_info.running_loss / len(self.dl_val))
        self.adv_loss_val_list.append(adv_epoch_info.running_loss / len(self.dl_val))
        self.cln_acc_val_list.append(cln_epoch_info.running_acc / len(self.dl_val))
        self.adv_acc_val_list.append(adv_epoch_info.running_acc / len(self.dl_val))
        
        # COCO evaluation
        print('\nVal CLEAN AP/AR')
        cln_all_APs, cln_mAP = self.ds_val.evaluate(
            cln_epoch_info.all_preds[:cln_epoch_info.idx], cln_epoch_info.all_boxes[:cln_epoch_info.idx], cln_epoch_info.image_paths[:cln_epoch_info.idx], res_folder=self.log_path,
            filename_prefix='cln_val')

        self.cln_mAP_val_list.append(cln_mAP)
        self.cln_APs_val_list.append(cln_all_APs)

        print('\nVal ADV AP/AR')
        adv_all_APs, adv_mAP = self.ds_val.evaluate(
            adv_epoch_info.all_preds[:adv_epoch_info.idx], adv_epoch_info.all_boxes[:adv_epoch_info.idx], adv_epoch_info.image_paths[:adv_epoch_info.idx], res_folder=self.log_path,
            filename_prefix='adv_val')

        self.adv_mAP_val_list.append(adv_mAP)
        self.adv_APs_val_list.append(adv_all_APs)

        print(f'Ep{self.epoch} - Clean Val Acc: {self.cln_acc_val_list[-1]:.3f} | Loss: {self.cln_loss_val_list[-1]:.5f} | AP: {self.cln_mAP_val_list[-1]:.3f}')
        print(f'                 Adv   Val Acc: {self.adv_acc_val_list[-1]:.3f} | Loss: {self.adv_loss_val_list[-1]:.5f} | AP: {self.adv_mAP_val_list[-1]:.3f}')
        print(f'                 Total Loss: {self.loss_val_list[-1]:.3f}')

        if self.use_tensorboard:
            self.summary_writer.add_scalar('val_loss', self.cln_loss_val_list[-1],
                                            global_step=self.epoch) # For clean
            self.summary_writer.add_scalar('val_acc', self.cln_acc_val_list[-1],
                                            global_step=self.epoch)# For clean
            self.summary_writer.add_scalar('val_mAP', self.cln_mAP_val_list[-1],
                                            global_step=self.epoch)# For clean
            
            self.summary_writer.add_scalar('Adv_val_loss', self.adv_loss_val_list[-1],
                                            global_step=self.epoch) # For clean
            self.summary_writer.add_scalar('Adv_val_acc', self.adv_acc_val_list[-1],
                                            global_step=self.epoch)# For clean
            self.summary_writer.add_scalar('Adv_val_mAP', self.adv_mAP_val_list[-1],
                                            global_step=self.epoch)# For clean
            
            self.summary_writer.add_scalar('total_val_loss', self.loss_val_list[-1],
                                            global_step=self.epoch)
            
            # if self.epoch % 10 == 0: 
            #     save_images(image, target, joints_target, output, joints_preds,
            #                 joints_data['joints_visibility'], self.summary_writer,
            #                 step=self.epoch, prefix='val_')
