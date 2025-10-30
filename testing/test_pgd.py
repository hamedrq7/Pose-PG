import argparse
import ast
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(1, os.getcwd())
from datasets.COCO import COCODataset
from training.COCO import COCOTrain
from models_.hrnet import HRNet
from models_.poseresnet import PoseResNet
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from misc.utils import flip_tensor, flip_back, get_final_preds
import datasets.CustomDS.COCO_configs as COCO_configs
from datasets.CustomDS.COCODataset import TopDownCocoDataset
from torch.utils.data import DataLoader

def load_model(model_name, model_c, model_nof_joints, model_bn_momentum, pretrained_weight_path, device): 
    if model_name == 'hrnet':
        model = HRNet(c=model_c, nof_joints=model_nof_joints,
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == 'poseresnet':
        model = PoseResNet(resnet_size=model_c, nof_joints=model_nof_joints, bn_momentum=model_bn_momentum).to(device)
    else:
        print('invalid model name')

def get_and_report(model, ds, loss_fn, X, y, y_targeted, joints_data, num_samples, model_nof_joints, ):
    all_preds = np.zeros((num_samples, model_nof_joints, 3), dtype=float)
    all_boxes = np.zeros((num_samples, 7), dtype=float)
    image_paths = []

    c = joints_data['center'].numpy()
    s = joints_data['scale'].numpy()
    score = joints_data['score'].numpy()
    pixel_std = 200  # ToDo Parametrize this
    bbox_id = None if not 'bbox_id' in joints_data.keys() else joints_data['bbox_id'].numpy()

    model.eval()

    output = model(X)
    loss = loss_fn(output, y, y_targeted)
    accs, avg_acc, cnt, joints_preds, joints_target = \
        ds.evaluate_accuracy(output, y)
    
    preds, maxvals = get_final_preds(True, output.detach(), c, s,
                                        pixel_std)  
    
    all_preds[:, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
    all_preds[:, :, 2:3] = maxvals.detach().cpu().numpy()
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * pixel_std, 1)
    all_boxes[:, 5] = score
    all_boxes[:, 6] = bbox_id
    image_paths.extend(joints_data['imgPath'])

    all_APs, mAP = ds.evaluate(
        all_preds[:num_samples], all_boxes[:num_samples], image_paths[:num_samples], res_folder='./')
    # all_APs, mAP = self.ds_train.evaluate_overall_accuracy(
    #     all_preds[:idx], all_boxes[:idx], image_paths[:idx], output_dir=self.log_path)
    
    print(f'Acc: {avg_acc.item():.3f} | Loss: {loss.item():.5f} | AP: {mAP:.3f}')


def main(exp_name,
         batch_size=2,
         num_workers=4,
         pretrained_weight_path=None,
         checkpoint_path=None,
         log_path='./logs',
         disable_tensorboard_log=False,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         disable_flip_test_images=False,
         image_resolution='(384, 288)',
         coco_root_path="./datasets/COCO",
         coco_bbox_path=None,
         seed=1,
         device=None,
         pre_trained_only=True,
         model_name = 'hrnet'
         ):

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True  # Enables cudnn
        torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

    # torch device
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    image_resolution = ast.literal_eval(image_resolution)


    # ds_val = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
    #                     data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_val_pipeline, dataset_info=COCO_configs.COCO_dataset_info, test_mode=False) # test_mode ? [?]
    
    ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/train2017/', 
                        data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_train_pipeline, dataset_info=COCO_configs.COCO_dataset_info, test_mode=False)

    dl = DataLoader(ds, batch_size=batch_size)

    # Load model
    model = load_model(model_name, model_c, model_nof_joints, model_bn_momentum, pretrained_weight_path, device)

    # Loss
    loss_fn = JointsMSELoss(False).to(device)

    # Load one batch
    data_iter = iter(dl)
    image, target, _, joints_data = next(data_iter)
    image = image.to(device)
    target = target.to(device)
    # target_weight = target_weight.to(device)
    
    ########################################### CLEAN
    X, y = Variable(image, requires_grad=True), Variable(target)

    get_and_report(model, ds, loss_fn, X, y, None, joints_data, batch_size, model_nof_joints)

    ################################ Adversarial
    EPSILON = 8/255.
    NUM_STEPS = 1
    STEP_SIZE = 1/255. 

    X_pgd = Variable(X.data, requires_grad=True)
    if False:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-EPSILON, EPSILON).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(NUM_STEPS):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            adv_output = model(X_pgd)
            adv_loss = loss_fn(adv_output, y, None)

        adv_loss.backward()
        eta = STEP_SIZE * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -EPSILON, EPSILON)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    get_and_report(model, ds, loss_fn, X_pgd, y, None, joints_data, batch_size, model_nof_joints)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=4)
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--checkpoint_path", "-c",
                        help="previous checkpoint path. Checkpoint will be loaded before training starts. It includes "
                             "the model, the optimizer, the epoch, and other parameters.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./')
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=50)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(256, 192)')
    parser.add_argument("--coco_root_path", help="COCO dataset root path", type=str, default="./datasets/COCO")
    parser.add_argument("--coco_bbox_path", help="path of detected bboxes to use during evaluation",
                        type=str, default=None)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--model_name", help="poseresnet or hrnet", type=str, default='poseresnet')
    
    args = parser.parse_args()

        
    main(**args.__dict__)
