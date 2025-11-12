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
import datasets.CustomDS.data_configs.COCO_configs as COCO_configs
from datasets.CustomDS.COCODataset import TopDownCocoDataset
from torch.utils.data import DataLoader

from misc.visualization import joints_dict
import matplotlib.pyplot as plt 
from datasets.CustomDS.eval_utils import pose_pck_accuracy, keypoints_from_heatmaps
from misc.general_utils import NormalizeByChannelMeanStd, set_seed_reproducability, get_device, perturb
from misc.log_utils import make_dir
import cv2 

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def get_joints(out, X, nof=17, res = (256, 192)): 
    boxes = np.asarray([0, 0, X.shape[2], X.shape[1]], dtype=float)
    # [x1, y1, x2, y2] 
    # print(boxes)
    
    # print(boxes.shape) # [4]
    
    pts = np.empty((out.shape[0], 3), dtype=np.float32)
    # [17, 3]
    # print(pts.shape)

    for j, joint in enumerate(out):
        pt = np.unravel_index(np.argmax(joint), (res[0] // 4, res[1] // 4))
        # print(pt) # (x, y)
        # print(joint.shape) # [target_h, target_w] 
        # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
        # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
        # 2: confidences
        pts[j, 0] = pt[0] * 1. / (res[0] // 4) * (boxes[3] - boxes[1]) + boxes[1]
        pts[j, 1] = pt[1] * 1. / (res[1] // 4) * (boxes[2] - boxes[0]) + boxes[0]
        pts[j, 2] = joint[pt]
        # print(boxes)
        # print(pt)
        # print(pts[j, 0])
        # print(pts[j, 1])
        # print(pts[j, 2])

    return pts 

def plot_joints(out, X, name, res, nof):
    
    print(out.shape) # [1, nof, target_h, target_w]
    print(X.shape) # [1, 3, H, W]
    boxes = np.repeat(
        np.asarray([[0, 0, X[0].shape[2], X[0].shape[1]]], dtype=float), len(X), axis=0
    )  # [x1, y1, x2, y2] 
    
    heatmaps = np.zeros((len(X), nof, res[0] // 4, res[1] // 4),
                                dtype=float)
    print(boxes.shape) # [1, 4]
    print(heatmaps.shape) # [1, nof, target_h, target_w]

    pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
    # [1, 17, 3]

    # For each human, for each joint: y, x, confidence
    for i, human in enumerate(out):
        heatmaps[i] = human # [17, t_h, t_w]
        for j, joint in enumerate(human):
            pt = np.unravel_index(np.argmax(joint), (res[0] // 4, res[1] // 4))
            # print(pt) # (x, y)
            # print(joint.shape) # [target_h, target_w] 
            # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
            # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
            # 2: confidences
            pts[i, j, 0] = pt[0] * 1. / (res[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
            pts[i, j, 1] = pt[1] * 1. / (res[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
            pts[i, j, 2] = joint[pt]
            # print(boxes[i])
            # print(pt)
            # print(pts[i, j, 0])
            # print(pts[i, j, 1])
            # print(pts[i, j, 2])
            
    pts = np.expand_dims(pts, axis=1)
    res = list()
    if False: # self.return_heatmaps:
        res.append(heatmaps)
    if False: # self.return_bounding_boxes:
        res.append(boxes)
    res.append(pts)

    joints = None
    if len(res) > 1:
        joints = res
    else:
        joints = res[0]

    ## probmalatic:
    joints=joints[0]

    fig = plt.figure(figsize=(90/2.54, 30/2.54))
    ax = fig.add_subplot(131)
    # ax.imshow((X[0]*std+mean).transpose(1, 2, 0))
    ax.imshow((X[0]).transpose(1, 2, 0))
    ax = fig.add_subplot(132)
    # ax.imshow((X[0]*std+mean).transpose(1, 2, 0))
    # ax.imshow((X[0]).transpose(1, 2, 0))
    ax.imshow(np.ones_like(X[0]).transpose(1, 2, 0))    

    bones = joints_dict()["coco"]["skeleton"]
    # bones = joints_dict()["mpii"]["skeleton"]

    for bone in bones:
        xS = [joints[:,bone[0],1], joints[:,bone[1],1]]
        yS = [joints[:,bone[0],0], joints[:,bone[1],0]]
        ax.plot(xS, yS, linewidth=3, c=(0,0.3,0.7))
    ax.scatter(joints[:,:,1],joints[:,:,0], s=20, c='r')

    # for heatmap
    ax = fig.add_subplot(133)
    ax.imshow(heatmaps[0].sum(0))

    plt.savefig(f'{name}.png')
    plt.clf()


def load_model(model_name, model_c, model_nof_joints, model_bn_momentum, pretrained_weight_path, device): 
    if model_name == 'hrnet':
        model = HRNet(c=model_c, nof_joints=model_nof_joints,
                        bn_momentum=model_bn_momentum).to(device)
    elif model_name == 'poseresnet':
        model = PoseResNet(resnet_size=model_c, nof_joints=model_nof_joints, bn_momentum=model_bn_momentum).to(device)
    else:
        print('invalid model name')

    if pretrained_weight_path is not None:
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(pretrained_weight_path, map_location=device),
            strict=False  # strict=False is required to load models pre-trained on imagenet
        )
        print('Pre-trained weights loaded.')
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print('Pre-trained weights missing keys:', missing_keys)
            print('Pre-trained weights unexpected keys:', unexpected_keys)
    return model

def get_and_report(model, ds, loss_fn, X, y, y_targeted, joints_data, num_samples, model_nof_joints, res=(256, 192), name='noname'):
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
    accs, avg_acc, cnt = pose_pck_accuracy(output.detach().cpu().numpy(), 
                                        y.detach().cpu().numpy(), 
                                        mask=y_targeted.detach().cpu().numpy().squeeze(-1) > 0,
                                        thr=0.05)
    
    preds, maxvals = keypoints_from_heatmaps(
        heatmaps=output.detach().cpu().numpy(),
        center=c, 
        scale=s,
        post_process="default", 
        kernel=11, # if ds.heatmap_sigma == 2. else 17, # carefull 
        target_type="GaussianHeatmap",
    ) 
    
    all_preds[:, :, 0:2] = preds[:, :, 0:2] # .detach().cpu().numpy()
    all_preds[:, :, 2:3] = maxvals # .detach().cpu().numpy()
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * pixel_std, 1)
    all_boxes[:, 5] = score
    all_boxes[:, 6] = bbox_id
    image_paths.extend(joints_data['imgPath'])

    # all_APs, mAP = ds.evaluate(
    #     all_preds[:num_samples], all_boxes[:num_samples], image_paths[:num_samples], res_folder='./')
    mAP = -1 

    return {
        'pck_acc': avg_acc.item(),
        'loss': loss.item(),
        'mAP': mAP,
        'preds': preds, 
        'maxvals': maxvals,
        'heatmaps': output.detach().cpu().numpy()
    }
    ### Visualization

    plot_joints(output[0].detach().cpu().numpy()[None, :, :, :], X[0].detach().cpu().numpy()[None, :, :, :], f'{name}_1', (256, 192), model_nof_joints)
    # plot_joints(output[1].detach().cpu().numpy()[None, :, :, :], X[1].detach().cpu().numpy()[None, :, :, :], f'{name}_2', (256, 192), model_nof_joints)

def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


def tensor2npimg(tensor, tr=True):
    if not tr:
        return bchw2bhwc(tensor[0].detach().cpu().numpy())
    else:
        return bchw2bhwc(tensor[0].detach().cpu().numpy()*std+mean)

def _show_images(img, advimg, enhance=127):
    np_img = tensor2npimg(img, False)
    np_advimg = tensor2npimg(advimg, False)
    np_perturb = np_advimg - np_img
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img)
    plt.axis("off")
    plt.title("original image")

    plt.subplot(1, 3, 2)
    plt.imshow(np_perturb * enhance)
    plt.axis("off")
    plt.title("the perturbation")
    
    plt.subplot(1, 3, 3)
    plt.imshow(np_advimg)
    plt.axis("off")
    plt.title("perturbed image")
    plt.savefig('pgd')
    plt.clf()

def main(
         pretrained_weight_path=None,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         image_resolution='(384, 288)',
         seed=1,
         device=None,
         model_name = 'hrnet',
         log_path='temp'
         ):

    # Seeds
    set_seed_reproducability(seed)

    # torch device
    device = get_device(device)

    image_resolution = ast.literal_eval(image_resolution)

    # pop out normalization pipeline val
    data_pipeline = COCO_configs.COCO_val_pipeline
    _ = data_pipeline.pop(3)

    # use diff joint weights to customize adv loss
    ds_info = COCO_configs.COCO_dataset_info
    ds_info['joint_weights'] = [
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ]
    # ds_info['joint_weights'] = [
    #     0., # nose
    #     0., # left_eye
    #     0., # right_eye
    #     0., # left_ear
    #     0., # right_ear
    #     0., # left_shoulder
    #     0., # right_shoulder
    #     1., # left_elbow
    #     1., # right_elbow
    #     0., # left_wrist
    #     0., # right_wrist
    #     0., # left_hip
    #     0., # right_hip
    #     0., # left_knee
    #     0., # right_knee
    #     0., # left_ankle
    #     0., # right_ankle
    # ]

    ds_cfg = COCO_configs.COCO_data_cfg
    ds_cfg['use_different_joint_weights'] = True
    # ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
    #                     data_cfg=ds_cfg, pipeline=data_pipeline, dataset_info=ds_info, test_mode=False,
    #                     indicies=None) # test_mode ? [?]
    
    # train but with val pipeline
    ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/train2017/', 
        data_cfg=ds_cfg, pipeline=data_pipeline, dataset_info=ds_info, test_mode=False,
        indicies = [761, 4377, 4554, 4711, 7277, 9797, 10082, 11292, 11655, 11826]
    )

    batch_size = 100
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)

    # Load model
    model = load_model(model_name, model_c, model_nof_joints, model_bn_momentum, pretrained_weight_path, device)
    normalizer = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to(device)
    model = nn.Sequential(normalizer, model) # Since normalization is not added

    # Loss
    loss_fn = JointsMSELoss(True).to(device)

    # Load one batch
    data_iter = iter(dl)
    image, target, target_weight, joints_data = next(data_iter)
    print(joints_data.keys())
    image = image.to(device)
    target = target.to(device)
    target_weight = target_weight.to(device)
    
    ########################################### CLEAN
    X, y, y_t = Variable(image, requires_grad=True), Variable(target), Variable(target_weight)

    EPSILON = 8/255.
    NUM_STEPS = 10
    STEP_SIZE = 2/255. 
    X_pgd = perturb(model, device, X, target, target_weight, loss_fn, EPSILON, NUM_STEPS, STEP_SIZE, rand_init=False)

    results_cln = get_and_report(model, ds, loss_fn, X, y, y_t, joints_data, X.shape[0], model_nof_joints, name='clean')
    results_adv = get_and_report(model, ds, loss_fn, X_pgd, y, y_t, joints_data, X_pgd.shape[0], model_nof_joints, name='adv')

    X = X.detach().cpu().numpy()
    X_pgd = X_pgd.detach().cpu().numpy()

    print('Plotting results')
    for idx in range(X.shape[0]): 
        # Directory_name = img_id followed by bbox_id
        sample_name = f'{joints_data['imgId'][idx].numpy()}_{joints_data['bbox_id'][idx].numpy()}'
        make_dir(f'{log_path}/{sample_name}')

        fig = plt.figure(figsize=(240, 60)) # figsize=(90/2.54, 30/2.54)
        
        # x
        ax = fig.add_subplot(241)
        ax.imshow(X[idx].transpose(1, 2, 0))
        ax.set_title('Clean')
        ax.axis("off")

        # joints
        ax = fig.add_subplot(242)
        ax.imshow(np.ones_like(X[idx]).transpose(1, 2, 0))
        cln_joints = get_joints(results_cln['heatmaps'][idx], X[idx])
        # print(cln_joints)
        # print(results_cln['preds'])
        bones = joints_dict()["coco"]["skeleton"]
        for bone in bones:
            xS = [cln_joints[bone[0],1], cln_joints[bone[1],1]]
            yS = [cln_joints[bone[0],0], cln_joints[bone[1],0]]
            ax.plot(xS, yS, linewidth=3, c=(0,0.3,0.7))
        ax.scatter(cln_joints[:,1],cln_joints[:,0], s=20, c='r')
        ax.set_title('Predicted Joints (cln)')
        ax.axis("off")

        # x+heatmap
        ax = fig.add_subplot(243)
        heatmap_up = cv2.resize(
            results_cln['heatmaps'][idx].sum(0),
            (192, 256),
            interpolation=cv2.INTER_LINEAR   # smooth upsampling
        )
        heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())
        overlay_intensity = 0.5  # 0–1
        ax.imshow(X[idx].transpose(1, 2, 0))
        ax.imshow(
            heatmap_up,
            cmap="jet",
            alpha=overlay_intensity
        )
        ax.set_title('Heatmaps (cln)')
        ax.axis("off")

        # perturbations
        ax = fig.add_subplot(244)
        pert = X_pgd[idx] - X[idx]
        ax.imshow(pert.transpose(1, 2, 0)* 50)
        ax.set_title('Perturbations (enhanced)')
        ax.axis("off")

        # x adv
        ax = fig.add_subplot(245)
        ax.imshow(X_pgd[idx].transpose(1, 2, 0))
        ax.set_title('Adv. Sample')
        ax.axis("off")

        # joints adv
        ax = fig.add_subplot(246)
        ax.imshow(np.ones_like(X_pgd[idx]).transpose(1, 2, 0))
        adv_joints = get_joints(results_adv['heatmaps'][idx], X_pgd[idx])
        # print(adv_joints)
        # print(results_adv['preds'])
        bones = joints_dict()["coco"]["skeleton"]
        for bone in bones:
            xS = [adv_joints[bone[0],1], adv_joints[bone[1],1]]
            yS = [adv_joints[bone[0],0], adv_joints[bone[1],0]]
            ax.plot(xS, yS, linewidth=3, c=(0,0.3,0.7))
        ax.scatter(adv_joints[:,1], adv_joints[:,0], s=20, c='r')
        ax.set_title('Predicted Joints (adv)')
        ax.axis("off")

        # x adv +heatmap
        ax = fig.add_subplot(247)
        heatmap_up = cv2.resize(
            results_adv['heatmaps'][idx].sum(0),
            (192, 256),
            interpolation=cv2.INTER_LINEAR   # smooth upsampling
        )
        heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())
        overlay_intensity = 0.5  # 0–1
        ax.imshow(X_pgd[idx].transpose(1, 2, 0))
        ax.imshow(
            heatmap_up,
            cmap="jet",
            alpha=overlay_intensity
        )
        ax.set_title('Heatmaps (adv)')
        ax.axis("off")

        # heatmap diff
        ax = fig.add_subplot(248)
        diff = results_adv['heatmaps'][idx] - results_cln['heatmaps'][idx]
        ax.imshow(diff.sum(0))
        ax.set_title('HM_adv-HM_cln')
        ax.axis("off")

        plt.savefig(f'{log_path}/{sample_name}/all.png')
        plt.clf( )
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./')
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=50)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(256, 192)')
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--model_name", help="poseresnet or hrnet", type=str, default='poseresnet')
    
    args = parser.parse_args()

        
    main(**args.__dict__)
