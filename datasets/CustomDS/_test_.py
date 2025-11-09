import sys 
import os 
sys.path.insert(1, os.getcwd())

from datasets.CustomDS.AnimalAP10KDataset import AnimalAP10KDataset
import datasets.CustomDS.data_configs.AP10K_configs as AP10K_configs
from torch.utils.data import DataLoader

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

import datasets.CustomDS.data_configs.COCO_configs as COCO_configs
from datasets.CustomDS.COCODataset import TopDownCocoDataset

train_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_train2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/train2017/', 
                        data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_train_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

# test_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
#                         data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_test_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

# val_ds = TopDownCocoDataset(f'{COCO_configs.COCO_data_root}/annotations/person_keypoints_val2017.json', img_prefix=f'{COCO_configs.COCO_data_root}/val2017/', 
#                         data_cfg=COCO_configs.COCO_data_cfg, pipeline=COCO_configs.COCO_val_pipeline, dataset_info=COCO_configs.COCO_dataset_info)

train_dl = DataLoader(train_ds, batch_size=16, num_workers=8)

import time
from tqdm import tqdm 

for i, (batch) in tqdm(train_dl):
    start = time.perf_counter()
    _ = batch  # just load, don't pass to model
    if i >= 20: break

print("Avg load time:", (time.perf_counter() - start)/20)

# for WHAT in val_dl:
#     print(type(WHAT))
#     print(WHAT.keys())

#     print(WHAT['img'].shape)
#     print(WHAT['target'].shape)
#     print(WHAT['target_weight'].shape)
    
#     print((WHAT['img_metas']['bbox_score']).shape)
    
#     break