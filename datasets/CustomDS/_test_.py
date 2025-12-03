import sys 
import os 
sys.path.insert(1, os.getcwd())

from torch.utils.data import DataLoader
from misc.general_utils import get_coco_loaders, get_rotation_coco


### 
ds_val = get_rotation_coco(image_resolution=[256, 192], model_name='poseresnet',
                            phase="train", test_mode=False, no_normalization = True) # test_mode should not be false here
dl = DataLoader(ds_val, batch_size=2)

import matplotlib.pyplot as plt
import os
import torch

save_dir = "saved_batches"
os.makedirs(save_dir, exist_ok=True)

max_steps = 10  # stop after this many batches
column_titles = ["HPE", "0-deg", "180-deg"]

for step, (batch_hpe, batch_rot_0, batch_rot_180) in enumerate(dl):
    if step >= max_steps:
        break

    B = 2
    assert B == 2, "Code assumes batch size = 2"

    X1, _, _, _ = batch_hpe
    X2, _, _, _ = batch_rot_0
    X3, _, _, _ = batch_rot_180
    
    # Stack into [B, 3, C, H, W]
    batch = torch.stack([X1, X2, X3], dim=1)

    fig, axes = plt.subplots(nrows=B, ncols=3, figsize=(9, 6))
    for col in range(3):
            axes[0, col].set_title(column_titles[col], fontsize=14)

    for i in range(B):
        for j in range(3):
            img = batch[i, j]

            # CHW → HWC for plotting
            img = img.permute(1, 2, 0).cpu()

            # Handle grayscale images
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[i, j].imshow(img, cmap="gray")
            else:
                axes[i, j].imshow(img)

            axes[i, j].axis("off")

    plt.tight_layout()

    # ---- Save to file ----
    filename = os.path.join(save_dir, f"batch_{step}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"Saved: {filename}")

exit()



pose_ds_val = get_coco_loaders(image_resolution=[256, 192], model_name="poseresnet",
                            phase="val", test_mode=False) # test_mode should not be false here

train_dl = DataLoader(pose_ds_val, batch_size=1)
it = iter(train_dl)
image, target, target_weight, joints_data = next(it)
print(image.shape)
exit()

# train_ds = AnimalAP10KDataset(f'{AP10K_configs.AP10K_data_root}/annotations/ap10k-train-split1.json', img_prefix=f'{AP10K_configs.AP10K_data_root}/data/', 
#                         data_cfg=AP10K_configs.AP10K_data_cfg, pipeline=AP10K_configs.AP10K_train_pipeline, dataset_info=AP10K_configs.AP10K_dataset_info)

# test_ds = AnimalAP10KDataset(f'{AP10K_configs.AP10K_data_root}/annotations/ap10k-test-split1.json', img_prefix=f'{AP10K_configs.AP10K_data_root}/data/', 
#                         data_cfg=AP10K_configs.AP10K_data_cfg, pipeline=AP10K_configs.AP10K_test_pipeline, dataset_info=AP10K_configs.AP10K_dataset_info)

# val_ds = AnimalAP10KDataset(f'{AP10K_configs.AP10K_data_root}/annotations/ap10k-val-split1.json', img_prefix=f'{AP10K_configs.AP10K_data_root}/data/', 
#                         data_cfg=AP10K_configs.AP10K_data_cfg, pipeline=AP10K_configs.AP10K_val_pipeline, dataset_info=AP10K_configs.AP10K_dataset_info)

# train_dl = DataLoader(train_ds, batch_size=2)

# for WHAT in train_dl:
#     print(type(WHAT))
#     print(WHAT.keys())

#     print(WHAT['img'].shape)
#     print(WHAT['target'].shape)
#     print(WHAT['target_weight'].shape)
    
#     print((WHAT['img_metas']['bbox_score']).shape)
    
#     break

# import datasets.CustomDS.data_configs.COCO_configs as COCO_configs
# from datasets.CustomDS.COCODataset import TopDownCocoDataset

# train_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/train2017/', 
#                         data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_train_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

# test_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
#                         data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_test_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

# val_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
#                         data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_val_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

# train_dl = DataLoader(train_ds, batch_size=16, num_workers=8)

# import time
# from tqdm import tqdm 

# i = 0
# for batch in tqdm(train_dl):
#     start = time.perf_counter()
#     _ = batch  # just load, don't pass to model
#     i += 1 
#     if i >= 2000: break

# print("Avg load time:", (time.perf_counter() - start)/2000)

# for WHAT in val_dl:
#     print(type(WHAT))
#     print(WHAT.keys())

#     print(WHAT['img'].shape)
#     print(WHAT['target'].shape)
#     print(WHAT['target_weight'].shape)
    
#     print((WHAT['img_metas']['bbox_score']).shape)
    
#     break

from misc.general_utils import get_coco_loaders
ds_val = get_coco_loaders(image_resolution=(256, 192), model_name='poseresnet',
                            phase="val", test_mode=False) # test_mode should not be false here
dl = DataLoader(ds_val, batch_size=2)

for step, (image, target, target_weight, joints_data) in enumerate(dl):
    print(image.shape)
    print(target.shape)
    print(target_weight.shape)
    print(joints_data.keys())
    break 
exit()

import datasets.CustomDS.data_configs.CrowdPose_configs as CrowdPoseConfigs
from datasets.CustomDS.CrowdPoseDataset import TopDownCrowdPoseDataset

train_ds = TopDownCrowdPoseDataset(
    ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_train.json',
    img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
    data_cfg=CrowdPoseConfigs.CrowdPose_data_cfg,
    pipeline=CrowdPoseConfigs.CrowdPose_train_pipeline,
    dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
    test_mode=False
)

val_ds = TopDownCrowdPoseDataset(
    ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_val.json',
    img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
    data_cfg=CrowdPoseConfigs.CrowdPose_data_cfg,
    pipeline=CrowdPoseConfigs.CrowdPose_train_pipeline,
    dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
    test_mode=False
)

test_ds = TopDownCrowdPoseDataset(
    ann_file=f'{CrowdPoseConfigs.CrowdPose_data_root}/annotations/crowdpose_test.json',
    img_prefix=f'{CrowdPoseConfigs.CrowdPose_data_root}/images/',
    data_cfg=CrowdPoseConfigs.CrowdPose_data_cfg,
    pipeline=CrowdPoseConfigs.CrowdPose_train_pipeline,
    dataset_info=CrowdPoseConfigs.CrowdPose_dataset_info,
    test_mode=False
)

dl = DataLoader(train_ds, batch_size=2)

for step, (image, target, target_weight, joints_data) in enumerate(dl):
    print(image.shape)
    print(target.shape)
    print(target_weight.shape)
    print(joints_data.keys())
    break 

dl = DataLoader(test_ds, batch_size=2)

for step, (image, target, target_weight, joints_data) in enumerate(dl):
    print(image.shape)
    print(target.shape)
    print(target_weight.shape)
    print(joints_data.keys())
    break 

dl = DataLoader(val_ds, batch_size=2)

for step, (image, target, target_weight, joints_data) in enumerate(dl):
    print(image.shape)
    print(target.shape)
    print(target_weight.shape)
    print(joints_data.keys())
    break 
