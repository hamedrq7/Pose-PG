import torch
import random 
import numpy as np 
from models_.poseresnet import PoseResNet
from models_.hrnet import HRNet

def get_model(model_name, model_c, model_nof_joints, model_bn_momentum, device, pretrained_weight_path=None): 
    # load model
    if model_name == 'hrnet':
        model = HRNet(c=model_c, nof_joints=model_nof_joints,
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == 'poseresnet':
        model = PoseResNet(resnet_size=model_c, nof_joints=model_nof_joints, 
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

    ### The models from MadyLab are a bit weird 
    if 'model' in checkpoint.keys() and 'optimizer' in checkpoint.keys() and 'schedule' in checkpoint.keys() and 'epoch' in checkpoint.keys() and len(checkpoint.keys()) == 4: 
        # https://www.dropbox.com/scl/fi/uwr6kbkchhi2t34czbzvh/imagenet_linf_8.pt?rlkey=fxnlz3irzmhvx8cbej7ye3fj5&st=l5msjf1p&dl=1

        state_dict = checkpoint["model"]

        # Remove "module.model." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if "attacker" not in k: 
                new_key = k.replace("module.model.", "")
                new_state_dict[new_key] = v
                # print(k, new_key)
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