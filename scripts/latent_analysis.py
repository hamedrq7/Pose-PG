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

    from misc.general_utils import get_coco_loaders
    ds_val = get_coco_loaders(image_resolution=image_resolution, model_name=model_name,
                                phase="val", test_mode=False)
    
    batch_size = 100
    dl = DataLoader(ds_val, batch_size=batch_size, num_workers=2)

    # Load model
    from misc.general_utils import get_model
    model = get_model(model_name, model_c, model_nof_joints, model_bn_momentum, pretrained_weight_path, device)
    
    data_iter = iter(dl)
    image, target, target_weight, joints_data = next(data_iter)
    print(joints_data.keys())
    image = image.to(device)
    target = target.to(device)
    target_weight = target_weight.to(device)
    
    feats, output = model(image, return_feats=True)

    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # feats has shape [N, C, H, W]

    # 1. Take the first element
    x = feats[0]     # shape: [C, H, W]

    C, H, W = x.shape

    # 2. Reshape to (H*W, C)
    samples = x.reshape(C, H * W).T   # shape: (H*W, C)

    # 3. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
    emb = tsne.fit_transform(samples)   # shape: (H*W, 2)

    # 4. Plot and SAVE the figure
    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.7)
    plt.title("t-SNE of feats[0] pixels")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.savefig("tsne_feats0.png", dpi=300, bbox_inches='tight')
    plt.close()

        
        
        
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
