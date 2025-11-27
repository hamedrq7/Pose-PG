import numpy as np
import torch
import torch.nn as nn 

from tqdm import tqdm
from datetime import datetime

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.Train import Train
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps
from training.COCO import COCO_standard_epoch_info

class SODEFTrain(Train):
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
        super(SODEFTrain, self).__init__(
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

    def run(self):
        """
        Runs the training.
        """

        print('\nTraining started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.reg1_all = []
        self.reg2_all = []
        self.reg3_all = []
        for ep in range(0, 0):
            self._phase2(ep)
        
        self._phase3_prep()

        # this is equivalent to phase 3 when you have called _phase3_prep() before 

        for self.epoch in range(10): # self.starting_epoch, self.epochs
            print('\nEpoch %d of %d @ %s' % (self.epoch + 1, self.epochs, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            #
            # Train

            self._train()

            #
            # Val

            self._val()

            #
            # LR Update

            if self.lr_decay:
                self.lr_scheduler.step()

            #
            # Checkpoint

            self._checkpoint()

        print('\nTraining ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def diag_jacobian(odefunc, z, t, device, exponent, trans):
        # z: (1, D) or any shape you want (flatten first if needed)

        v = torch.randn_like(z)        # random vector
        v = v / (v.norm() + 1e-8)

        f = odefunc(t, z)              # forward
        Jv = torch.autograd.grad(f, z, v, create_graph=True)[0]   # Jv product

        # diag(J) ≈ v * Jv (elementwise)
        diagJ = v * Jv                 
        
        regu_diag = torch.exp(exponent * (diagJ + trans)).mean()
        return regu_diag

    def df_dz_regularizer(self, odefunc, z, time_df, numm, device, exponent, trans, exponent_off, transoffdig):
        regu_diag = 0.
        regu_offdiag = 0.0
        for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
            batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
            batchijacobian = batchijacobian.view(z.shape[1],-1)
            if batchijacobian.shape[0]!=batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")
                
            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent*(tempdiag+trans))
            offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
            off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
            regu_offdiag += off_diagtemp

        print('diag mean: ',tempdiag.mean().item())
        print('offdiag mean: ',offdiat.mean().item())
        return regu_diag/numm, regu_offdiag/numm
    
    def f_regularizer(self, odefunc, z, time_df, device, exponent_f):
        tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
        regu_f = torch.pow(exponent_f*tempf,2)
        print('tempf: ', tempf.mean().item())
        return regu_f
    
    def temp1(odefunc, z, numm, time_df, device, exponent, trans, exponent_off, transoffdig, ):
        regu_diag = 0.
        regu_offdiag = 0.0
        for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
            batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
            batchijacobian = batchijacobian.view(z.shape[1],-1)
            if batchijacobian.shape[0]!=batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")
            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent*(tempdiag+trans))
            offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
            off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
            regu_offdiag += off_diagtemp

        print('diag mean: '+str(tempdiag.mean().item())+'\n')
        print('offdiag mean: '+str(offdiat.mean().item())+'\n')
        return 0
    def temp2(odefunc, z, time_df, device, exponent_f):
        tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
        regu_f = torch.pow(exponent_f*tempf,2)
        print('tempf: '+str(tempf.mean().item())+'\n')
        return 0


    def _phase2(self, epoch): 
        """
        Freeze the FE and the FC, only train ODE block with regularizers
        """
        
        weight_diag = 10 # eq 4
        weight_offdiag = 0 # eq 5
        weight_f = 0.1 # eq 3

        weight_norm = 0
        weight_lossc =  0

        exponent = 1.0
        exponent_off = 0.1 
        exponent_f = 50
        time_df = 1
        trans = 1.0
        transoffdig = 1.0
        numm = 16


        phase2optim = torch.optim.Adam(self.model.ode_layer.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

        self.model.train()
        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)
            
            phase2optim.zero_grad()
            feats = self.model.phase2_forward(image).detach()
            feats.requires_grad_(True)

            regu1, regu2  = self.df_dz_regularizer(self.model.odefunc, feats, time_df, numm, self.device, exponent, trans, exponent_off, transoffdig)
            regu1 = regu1.mean() # eq 4
            regu2 = regu2.mean() # eq 5
            print("regu1:weight_diag "+str(regu1.item())+':'+str(weight_diag))
            print("regu2:weight_offdiag "+str(regu2.item())+':'+str(weight_offdiag))
            regu3 = self.f_regularizer(self.model.odefunc, feats, time_df, self.device, exponent_f) # eq 3
            regu3 = regu3.mean()
            print("regu3:weight_f "+str(regu3.item())+':'+str(weight_f))
            loss = weight_f*regu3 + weight_diag*regu1+ weight_offdiag*regu2

            loss.backward()
            phase2optim.step()
            
            if self.use_tensorboard:
                self.summary_writer.add_scalar('reg1', regu1.item(),
                                                global_step=epoch*len(self.dl_train)+step)
                self.summary_writer.add_scalar('regu2', regu2.item(),
                                                global_step=epoch*len(self.dl_train)+step)
                self.summary_writer.add_scalar('regu3', regu3.item(),
                                                global_step=epoch*len(self.dl_train)+step)
            
            # if step % 1 == 0:
            #     with torch.no_grad():
            #         temp1(odefunc, y00, text_file)
            #         temp2(odefunc, y00, text_file)

    def _phase3_prep(self): 
        "train both ode and the Final layer with normal HPE loss"
        
        # Should I intialize final layer randomly or use previous init? 
        self.model.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=self.model_nof_joints,
            kernel_size=1,
            stride=1,
            padding=0
        ).to(self.device)

        self.model.zero_grad()

        trainable_names = ["ode_layer", "final_layer"]

        for name, param in self.model.named_parameters():
            # if this parameter belongs to layers you want to keep trainable
            if any(tn in name for tn in trainable_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.phase3optim = torch.optim.Adam([
            {"params": self.model.ode_layer.parameters(), 'lr': 1e-5, 'eps':1e-6,},
            {"params": self.model.final_layer.parameters(), 'lr': 1e-2, 'eps':1e-4,}
        ], amsgrad=True)

    def _train(self):

        epoch_info = COCO_standard_epoch_info(epoch=-1, phase="train", num_samples=len(self.ds_train), model_nof_joints=self.model_nof_joints)

        self.model.train()

        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.phase3optim.zero_grad()
            output = self.model(image) # [N, K, H, W] output heatmaps
            loss = self.loss_fn(output, target, target_weight) # MSE loss
            loss.backward()
            self.phase3optim.step()

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
