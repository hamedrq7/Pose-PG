import torch
import torch.nn as nn 
import random 
import numpy as np 
from models_.poseresnet import PoseResNet
from models_.hrnet import HRNet
from models_.vitpose import VitPose
from models_.poseresnet_sodef import PoseResNet_SODEF
from models_.poseresnet_ood import PoseResNetOOD
from misc.checkpoint import load_checkpoint
from losses.loss import JointsMSELoss, JointsOHKMMSELoss

from torch.autograd import Variable
import torch.optim as optim


def get_imagenet_loaders(image_resolution, phase: str, no_normalization: bool = False):
    W, H = image_resolution[1], image_resolution[0]
    
    from torchvision import transforms
    import datasets.CustomDS.data_configs.ImageNet_configs as configs 

    if phase == "val": 
        print('ImageNet Validation set, only animals...')
        
        val_transform = transforms.Compose([
            transforms.Resize(H),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_ds = configs.FilteredImageNet(
            root=configs.root,
            split="val",
            wnid_file=configs.animal_winds_path,
            transform=val_transform
        )

        print(len(val_ds))       # number of animal images
        print(len(val_ds.classes))   # should be 398

    elif phase == "train": 
        print('ImageNet Training set, only animals...')
        
        val_transform = transforms.Compose([
            transforms.Resize(H),
            transforms.CenterCrop((H, W)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_ds = configs.FilteredImageNet(
            root=configs.root,
            split="val",
            wnid_file=configs.animal_winds_path,
            transform=val_transform
        )

        print(len(val_ds))       # number of animal images
        print(len(val_ds.classes))   # should be 398

    ###### Train
    train_pipeline = [LoadImageFromFile()]
    train_pipeline.append(TopDownRandomFlip(flip_prob=0.5))
    train_pipeline.append(TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3))
    train_pipeline.append(TopDownGetRandomScaleRotation(rot_factor=45., scale_factor=0.35))
    train_pipeline.append(TopDownAffine(use_udp=udp))

    ###### Val
    val_pipeline = [LoadImageFromFile()]
    val_pipeline.append(TopDownAffine(use_udp=udp))
    val_pipeline.append(ToTensor())
    if not no_normalization:
        val_pipeline.append(NormalizeTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ))



def get_coco_loaders(image_resolution, model_name, phase: str, test_mode: bool, no_normalization: bool = False):
    import datasets.CustomDS.data_configs.COCO_configs as COCO_configs
    from datasets.CustomDS.COCODataset import TopDownCocoDataset

    tr_ppl, val_ppl, te_ppl = COCO_configs.get_pipelines(image_resolution=image_resolution, model_name=model_name, no_normalization=no_normalization)

    if phase == 'train': 
        ds_train = TopDownCocoDataset(
            ann_file=f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json',
            img_prefix=f'{COCO_configs.COCO_data_root}/train2017/',
            data_cfg=COCO_configs.get_data_cfg(image_resolution=image_resolution),
            pipeline=tr_ppl,
            dataset_info=COCO_configs.COCO_dataset_info,
            test_mode=test_mode, 
        )
        return ds_train
    elif phase == "val": 
        ds_val = TopDownCocoDataset(
            ann_file=f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json',
            img_prefix=f'{COCO_configs.COCO_data_root}/val2017/',
            data_cfg=COCO_configs.get_data_cfg(image_resolution=image_resolution),
            pipeline=val_ppl,
            dataset_info=COCO_configs.COCO_dataset_info,
            test_mode=test_mode,  
        )
        return ds_val
    

def get_crowdpose_loaders(image_resolution, model_name, phase: str, test_mode: bool, no_normalization: bool = False):
    import datasets.CustomDS.data_configs.CrowdPose_configs as CrowdPoseConfigs
    from datasets.CustomDS.CrowdPoseDataset import TopDownCrowdPoseDataset

    tr_ppl, val_ppl, te_ppl = CrowdPoseConfigs.get_pipelines(image_resolution=image_resolution, model_name=model_name, no_normalization=no_normalization)

    if phase == 'train': 
        ds_train = TopDownCrowdPoseDataset(
            ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_train.json',
            img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
            data_cfg=CrowdPoseConfigs.get_data_cfg(image_resolution=image_resolution),
            pipeline=tr_ppl,
            dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
            test_mode=test_mode
        )
        return ds_train
    elif phase == "val": 
        ds_val = TopDownCrowdPoseDataset(
            ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_val.json',
            img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
            data_cfg=CrowdPoseConfigs.get_data_cfg(image_resolution=image_resolution),
            pipeline=val_ppl,
            dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
            test_mode=test_mode
        )
        return ds_val
    elif phase == "test": 
        ds_test = TopDownCrowdPoseDataset(
            ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_test.json',
            img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
            data_cfg=CrowdPoseConfigs.get_data_cfg(image_resolution=image_resolution),
            pipeline=te_ppl,
            dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
            test_mode=test_mode
        )
        return ds_test


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

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def get_loss_fn(loss: str, device, use_target_weight=True):
    # define loss
    if loss == 'JointsMSELoss':
        loss_fn = JointsMSELoss(use_target_weight).to(device)
    elif loss == 'JointsOHKMMSELoss':
        loss_fn = JointsOHKMMSELoss(use_target_weight).to(device)
    elif loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError
    return loss_fn


class ReIndexWrapper(nn.Module):
    def __init__(self, model, index_map):
        super().__init__()
        self.model = model
        self.index_map = index_map
    
    def forward(self, input):
        normal_input = self.model(input)
        return normal_input[:, self.index_map, :, :]

def re_index_model_output(model, index_map): 
    # Assuming model is trained on COCO: 
    """
    For COCO -> Ap10K, a naive way is this
    index_map = [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16]
    reordered_output = output[:, index_map, :, :]
    """
    if index_map == "ap10k": 
        print('Re indexing output channels of model, only use for NAIVE zero shot testing from COCO to AP10K')
        return ReIndexWrapper(model, [2, 0, 1, 3, 4, 5, 8, 6, 9, 7, 10, 11, 14, 12, 15, 13, 16])
    elif index_map == "crowdpose": 
        print("Reindexing for COCO-CrowdPose")
        return ReIndexWrapper(model, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0])

def get_model(model_name, model_c, model_nof_joints, model_bn_momentum, device, pretrained_weight_path=None): 
    # load model
    if model_name == 'hrnet':
        model = HRNet(c=model_c, nof_joints=model_nof_joints,
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == 'poseresnet':
        model = PoseResNet(resnet_size=model_c, nof_joints=model_nof_joints, 
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == "vitpose_small":
        model = VitPose()
    elif model_name == "poseresnet_sodef": 
        model = PoseResNet_SODEF(resnet_size=model_c, nof_joints=model_nof_joints, 
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == "poseresnet_ood":
        model = PoseResNetOOD(resnet_size=model_c, nof_joints=model_nof_joints, 
                        bn_momentum=model_bn_momentum).to(device)
    if not pretrained_weight_path is None:
        model = load_pretrained(model, pretrained_weight_path, device=device)
    
    return model.to(device)
    
def load_pretrained(model, pretrained_weight_path, device):
    checkpoint = torch.load(pretrained_weight_path, map_location=device, weights_only=False) 

    #### TODO [ ]
    ####### From robustbench: 
    # https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/imagenet.py
    
    #### HRNET REPO, standard r50
    # resnet50 from https://drive.google.com/drive/folders/1E6j6W7RqGhW1o7UHgiQ9X4g8fVJRU9TX
    # Its from the hrnet repo: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/tree/master
    # pretrained_weight_path = 'C:/Users/hamed/Downloads/resnet50-19c8e357.pth'

    ## Models From VitPose repo: 
    if 'state_dict' in checkpoint.keys() and len(checkpoint.keys()) == 1: 
        new_state_dict = checkpoint['state_dict']

    ### The models from MadyLab are a bit weird 
    elif 'model' in checkpoint.keys() and 'optimizer' in checkpoint.keys() and 'schedule' in checkpoint.keys() and 'epoch' in checkpoint.keys():
        if len(checkpoint.keys()) == 4 or ('amp' in checkpoint.keys() and len(checkpoint.keys()) == 5): 
        # https://www.dropbox.com/scl/fi/uwr6kbkchhi2t34czbzvh/imagenet_linf_8.pt?rlkey=fxnlz3irzmhvx8cbej7ye3fj5&st=l5msjf1p&dl=1
            print('Model trained from MadryLab...')
            
            state_dict = checkpoint["model"]

            # Remove "module.model." prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if "attacker" not in k: 
                    new_key = k.replace("module.model.", "")
                    new_state_dict[new_key] = v
                    # print(k, new_key)
    
    ### Models from AP10-K repo (linked in their readme)
    elif 'meta' in checkpoint.keys() and 'mmcv_version' in checkpoint['meta']: 
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if "backbone." in k: 
                new_key = k.replace("backbone.", "")
            elif "keypoint_head." in k: 
                new_key = k.replace("keypoint_head.", "")
            else:
                print('error in loading model weights (name matchin issue)')
                return None
            new_state_dict[new_key] = v
            
    ### For chackpoints saved using your own code: 
    elif 'epoch' in checkpoint.keys() and 'model' in checkpoint.keys() and 'optimizer' in checkpoint.keys() and 'params' in checkpoint.keys(): 
        print('Model is trained using our own code')
        # epoch, model, optimizer, params = load_checkpoint(pretrained_weight_path, model, device=device)
        
        # Trained Adversarially: 
        if '0.mean' in checkpoint['model'].keys() and '0.std' in checkpoint['model'].keys():
            new_state_dict = {}
            for k, v in checkpoint['model'].items():
                if k.startswith("1."):
                    new_key = k[2:]  # remove the first two characters ("1.")
                else:
                    new_key = k
                new_state_dict[new_key] = v
                # print(k, new_key)
        else:
            # normal training
            new_state_dict = checkpoint['model']

    else: 
        new_state_dict = checkpoint
        
    missing_keys, unexpected_keys = model.load_state_dict(
        # torch.load(pretrained_weight_path, map_location=device, weights_only = False),
        new_state_dict,
        strict=False,  # strict=False is required to load models pre-trained on imagenet
    )
    print('Pre-trained weights loaded.')
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print('Pre-trained weights missing keys:', missing_keys)
        print('Pre-trained weights unexpected keys:', unexpected_keys)
    else:
        print('All pretrained weights keys matched')
    
    return model 

def set_seed_reproducability(seed): 
    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True  # Enables cudnn
        torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

def get_device(device):
    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
    
    print('device: ', device)
    return device