import os
import sys
from datetime import datetime

import tensorboardX as tb
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from misc.checkpoint import save_checkpoint, load_checkpoint, save_results, save_results_modular
from misc.utils import flip_tensor, flip_back
from misc.visualization import save_images
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet
from misc.log_utils import Logger 
from misc.general_utils import get_model, get_device, get_loss_fn

class TrainAuxilary(object):
    """
    Train  class.

    The class provides a basic tool for training HRNet.
    Most of the training options are customizable.

    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 exp_name,
                 pose_ds_train,
                 pose_ds_val,
                 aux_ds_train,
                 aux_ds_val,
                 aux_loss_weight: float, 
                 aux_loss: str, 
                 epochs=210,
                 pose_batch_size=16,
                 aux_batch_size=16,
                 num_workers=8,
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
                 model_name='poseresnet',
                 model_c=48,
                 model_nof_joints=17,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None,
                 ):
        """
        Add 
        """
        super(TrainAuxilary, self).__init__()

        self.exp_name = exp_name
        self.pose_ds_train = pose_ds_train
        self.pose_ds_val = pose_ds_val
        self.aux_ds_train = aux_ds_train
        self.aux_ds_val = aux_ds_val
        self.aux_loss = aux_loss
        self.aux_loss_weight = aux_loss_weight 
        self.epochs = epochs
        self.pose_batch_size = pose_batch_size
        self.aux_batch_size = aux_batch_size
        self.num_workers = num_workers
        self.pose_loss = pose_loss
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.pretrained_weight_path = pretrained_weight_path
        self.checkpoint_path = checkpoint_path
        self.log_path = os.path.join(log_path, self.exp_name)
        self.use_tensorboard = use_tensorboard
        self.model_name = model_name
        self.model_c = model_c
        self.model_nof_joints = model_nof_joints
        self.model_bn_momentum = model_bn_momentum
        self.flip_test_images = flip_test_images
        self.epoch = 0

        self.device = get_device(device)
        
        os.makedirs(self.log_path, 0o755, exist_ok=False)  # exist_ok=False to avoid overwriting        
        #
        # write all experiment parameters in parameters.txt and in tensorboard text field
        if self.use_tensorboard:
            self.summary_writer = tb.SummaryWriter(self.log_path)
            
        self.parameters = [x + ': ' + str(y) + '\n' for x, y in locals().items()]
        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        if self.use_tensorboard:
            self.summary_writer.add_text('parameters', '\n'.join(self.parameters))

        sys.stdout = Logger("{}/{}/run.log".format(log_path, exp_name))
        # sys.stderr = sys.stdout
        command_line_args = sys.argv
        command = " ".join(command_line_args)
        print(f"The command that ran this script: {command}")

        self.model = get_model(model_name=self.model_name, model_c=self.model_c, model_nof_joints=self.model_nof_joints, 
                          model_bn_momentum=self.model_bn_momentum, device=self.device, pretrained_weight_path=self.pretrained_weight_path)

        # load previous checkpoint
        if self.checkpoint_path is not None:
            print('Loading checkpoint %s...' % self.checkpoint_path)
            if os.path.isdir(self.checkpoint_path):
                path = os.path.join(self.checkpoint_path, 'checkpoint_last.pth')
            else:
                path = self.checkpoint_path
            self.starting_epoch, self.model, self.optim, self.params = load_checkpoint(path, self.model, self.optim,
                                                                                       self.device)
        else:
            self.starting_epoch = 0

        self.pose_loss_fn = get_loss_fn(self.pose_loss, self.device)
        self.aux_loss_fn = get_loss_fn(self.aux_loss, self.device)

        if optimizer == 'SGD':
            self.optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                             momentum=self.momentum, nesterov=self.nesterov)
        elif optimizer == 'Adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if lr_decay:
            self.lr_scheduler = MultiStepLR(self.optim, list(self.lr_decay_steps), gamma=self.lr_decay_gamma,
                                            last_epoch=self.starting_epoch if self.starting_epoch else -1)

        # load train and val datasets
        self.pose_dl_train = DataLoader(self.pose_ds_train, batch_size=self.pose_batch_size, shuffle=True,
                                   num_workers=self.num_workers, drop_last=False, pin_memory=True)
        self.pose_len_dl_train = len(self.pose_dl_train)
        self.pose_dl_val = DataLoader(self.pose_ds_val, batch_size=self.pose_batch_size, shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        self.pose_len_dl_val = len(self.pose_dl_val)

        # Aux dataloader
        self.aux_dl_train, self.aux_len_dl_train, self.aux_dl_val, self.aux_len_dl_val = self._create_aux_dataloaders()

        self.best_total_loss = None
        self.best_pose_loss = None
        self.best_aux_loss = None
        self.best_aux_acc = None
        self.best_pose_acc = None
        self.best_pose_mAP = None

        self.total_loss_train_list = []   
        self.total_loss_val_list = []  
        self.pose_loss_train_list = []  
        self.pose_loss_val_list = []  
        self.aux_loss_train_list = []   
        self.aux_loss_val_list = []  
        
        self.pose_acc_train_list = []  
        self.pose_acc_val_list = []
        self.pose_mAP_train_list = []  
        self.pose_mAP_val_list = [] 
        self.pose_APs_train_list = []  
        self.pose_APs_val_list = [] 
        
    def _get_base_results_dict(self): 
        return {
            'total_loss_train_list': self.total_loss_train_list,
            'total_loss_val_list': self.total_loss_val_list,
            'pose_loss_train_list': self.pose_loss_train_list,
            'pose_loss_val_list': self.pose_loss_val_list,
            'aux_loss_train_list': self.aux_loss_train_list,
            'aux_loss_val_list': self.aux_loss_val_list,
            'pose_acc_train_list': self.pose_acc_train_list,
            'pose_acc_val_list': self.pose_acc_val_list,
            'pose_mAP_train_list': self.pose_mAP_train_list,
            'pose_mAP_val_list': self.pose_mAP_val_list,
            'pose_APs_train_list': self.pose_APs_train_list,
            'pose_APs_val_list': self.pose_APs_val_list,
        }
    
    def _create_aux_dataloaders(self): 
        return 
        
    def _get_results_dict(self): 
        return
    
    def _train(self):
        return 
    
    def _val(self):
        return 
    
    def _checkpoint(self):
        # save_results(os.path.join(self.log_path, 'results.npz'), self.acc_train_list, self.loss_train_list, self.mAP_train_list, self.APs_train_list, self.acc_val_list, self.loss_val_list, self.mAP_val_list, self.APs_val_list)
        save_results_modular(
            os.path.join(self.log_path, 'results.npz'),
            self._get_results_dict(self)
        )
        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, model=self.model,
                        optimizer=self.optim, params=self.parameters)

        if self.best_pose_loss is None or self.best_pose_loss > self.pose_loss_val_list[-1]:
            self.best_pose_loss = self.pose_loss_val_list[-1]
            print('best_pose_loss %f at epoch %d' % (self.best_pose_loss, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_pose_loss.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
            
        if self.best_pose_acc is None or self.best_pose_acc < self.pose_acc_val_list[-1]:
            self.best_pose_acc = self.pose_acc_val_list[-1]
            print('best_pose_acc %f at epoch %d' % (self.best_pose_acc, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_pose_acc.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
        if self.best_pose_mAP is None or self.best_pose_mAP < self.pose_mAP_val_list[-1]:
            self.best_pose_mAP = self.pose_mAP_val_list[-1]
            print('best_pose_mAP %f at epoch %d' % (self.best_pose_mAP, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_pose_mAP.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)

    def run(self):
        """
        Runs the training.
        """

        print('\nTraining started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # start training
        for self.epoch in range(self.starting_epoch, self.epochs):
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
