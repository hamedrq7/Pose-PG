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
    
    batch_size = 10
    dl = DataLoader(ds_val, batch_size=batch_size, num_workers=2)

    # Load model
    from misc.general_utils import get_model
    model = get_model(model_name, model_c, model_nof_joints, model_bn_momentum, device, pretrained_weight_path)
    
    data_iter = iter(dl)
    image, target, target_weight, joints_data = next(data_iter)
    print(joints_data.keys())
    image = image.to(device)
    target = target.to(device)
    target_weight = target_weight.to(device)
    
    feats, output = model(image, return_feats=True)

    feats = feats.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    # import numpy as np
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt

    # # feats: [N, C, H, W]
    # N, C, H, W = feats.shape

    # # 1. reshape to (N*H*W, C)
    # samples = feats.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)

    # # 2. run t-SNE
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', verbose=1, init='pca')
    # emb = tsne.fit_transform(samples)   # shape: (N*H*W, 2)

    # # 3. simple 2D scatter
    # plt.figure(figsize=(6, 6))
    # plt.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.6)
    # plt.title("t-SNE of feats reshaped to (N*H*W, C)")
    # plt.xlabel("Dim 1")
    # plt.ylabel("Dim 2")
    # plt.savefig("tsne_feats_all.png", dpi=300, bbox_inches='tight')
    # plt.close()


    import numpy as np
    import matplotlib.pyplot as plt

    # target__ shape: [17, H, W]
    target__ = target.cpu().detach().numpy()[0]
    # Sum over the first axis → shape becomes [H, W]
    heat = target__.sum(axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(heat, cmap='hot')
    plt.colorbar()
    plt.title("Summed target heatmap")
    plt.savefig("target_sum_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    import numpy as np
    import matplotlib.pyplot as plt

    # feats: [N, C, H, W]

    x = feats[0]        # [C, H, W]
    C, H, W = x.shape

    # Average over channels → [H, W]
    avg_map = x.mean(axis=0)

    # Plot heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_map, cmap='viridis')
    plt.colorbar()
    plt.title("Channel-averaged heatmap")
    plt.savefig("avg_channels_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # feats: [N, C, H, W]

    # take first element
    x = feats[0]    # [C, H, W]
    C, H, W = x.shape

    # flatten to (H*W, C)
    samples = x.reshape(C, H*W).T

    # run t-SNE with 1 dimension
    tsne = TSNE(n_components=1, perplexity=30, learning_rate='auto', init='pca')
    emb1 = tsne.fit_transform(samples)     # shape: (H*W, 1)

    # reshape back to (H, W)
    tsne_map = emb1.reshape(H, W)

    # plot as heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(tsne_map, cmap='viridis')
    plt.colorbar()
    plt.title("1D t-SNE heatmap of feats[0]")
    plt.savefig("tsne_1d_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    ########################################## Load some fron ImageNet 
    imagenet_root = 'imagenet'
    # ----------- DATA -----------
    from torchvision import datasets, transforms, models
    import torchvision 
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256, 192),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    ds = torchvision.datasets.ImageNet(f'{imagenet_root}', split='val', transform=val_transform)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    data_iter = iter(dl)
    image, label = next(data_iter)
    image = image.to(device)
    label = label.to(device)

    feats, output = model(image, return_feats=True)
    feats = feats.cpu().detach().numpy()

    import numpy as np
    import matplotlib.pyplot as plt

    # feats: [N, C, H, W]
    x = feats[0]        # [C, H, W]
    C, H, W = x.shape

    # Average over channels → [H, W]
    avg_map = x.mean(axis=0)

    # Plot heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_map, cmap='viridis')
    plt.colorbar()
    plt.title("ImageNet Channel-averaged heatmap")
    plt.savefig("ImageNet avg_channels_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # feats: [N, C, H, W]

    # take first element
    x = feats[0]    # [C, H, W]
    C, H, W = x.shape

    # flatten to (H*W, C)
    samples = x.reshape(C, H*W).T

    # run t-SNE with 1 dimension
    tsne = TSNE(n_components=1, perplexity=30, learning_rate='auto', init='pca')
    emb1 = tsne.fit_transform(samples)     # shape: (H*W, 1)

    # reshape back to (H, W)
    tsne_map = emb1.reshape(H, W)

    # plot as heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(tsne_map, cmap='viridis')
    plt.colorbar()
    plt.title("ImageNet 1D t-SNE heatmap of feats[0]")
    plt.savefig("ImageNet tsne_1d_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    import torchvision.transforms.functional as F
    img_vis = F.to_pil_image(
        image[0].cpu().detach() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) +
            torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    )

    # Plot
    plt.imshow(img_vis)
    plt.title(f"Class: {ds.classes[label]}")
    plt.axis("off")
    plt.show()

    # Save
    img_vis.save("sample_imagenet.png")
    print("Saved to sample_imagenet.png")

    
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
