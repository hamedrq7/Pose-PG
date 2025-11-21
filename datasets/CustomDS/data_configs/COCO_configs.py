
COCO_data_root = 'datasets/COCO'

######## From configs\_base_\datasets\coco.py ######## 
COCO_dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])

######## ### From configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\hrnet_w32_coco_256x192.py

COCO_channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])


from datasets.CustomDS.augmentaions import (
    LoadImageFromFile, 
    TopDownRandomFlip,
    TopDownHalfBodyTransform,
    TopDownGetRandomScaleRotation,
    TopDownAffine,
    ToTensor,
    NormalizeTensor,
    TopDownGenerateTarget,
    Collect
)

def get_data_cfg(image_resolution):
    W, H = image_resolution[1], image_resolution[0]
    heatmap_size = [W//4, H//4]
    if (W==192 and H==256):
        heatmap_sigma = 2.0 
    elif (W==288 and H==384):
        heatmap_sigma = 3.0

    COCO_data_cfg = dict(
        image_size=[W, H],
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        num_output_channels=COCO_channel_cfg['num_output_channels'],
        num_joints=COCO_channel_cfg['dataset_joints'],
        dataset_channel=COCO_channel_cfg['dataset_channel'],
        inference_channel=COCO_channel_cfg['inference_channel'],
        soft_nms=False,
        nms_thr=1.0,
        oks_thr=0.9,
        vis_thr=0.2,
        use_gt_bbox=False,
        det_bbox_thr=0.0,
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        use_different_joint_weights=False
    )

    return COCO_data_cfg

def get_pipelines(image_resolution, model_name, no_normalization: bool = False):
    data_cfg = get_data_cfg(image_resolution)

    if model_name == 'poseresnet' or model_name == 'hrnet':
        udp = False
    elif model_name == 'vitpose_small':
        udp = True 

    ###### Train
    train_pipeline = [LoadImageFromFile()]
    train_pipeline.append(TopDownRandomFlip(flip_prob=0.5))
    train_pipeline.append(TopDownHalfBodyTransform(num_joints_half_body=8, prob_half_body=0.3))
    train_pipeline.append(TopDownGetRandomScaleRotation(rot_factor=45., scale_factor=0.35))
    train_pipeline.append(TopDownAffine(use_udp=udp))
    train_pipeline.append(ToTensor())
    if not no_normalization: 
        train_pipeline.append(NormalizeTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]))
    train_pipeline.append(TopDownGenerateTarget(sigma=data_cfg['heatmap_sigma'], encoding='UDP' if udp else 'MSRA')) 
    train_pipeline.append(Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'img_id', 'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'
        ]
    ))

    ###### Val
    val_pipeline = [LoadImageFromFile()]
    val_pipeline.append(TopDownAffine())
    val_pipeline.append(ToTensor())
    if not no_normalization:
        val_pipeline.append(NormalizeTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ))
    val_pipeline.append(TopDownGenerateTarget(sigma=data_cfg['heatmap_sigma'], encoding='UDP' if udp else 'MSRA')) #### [?] 
    val_pipeline.append(Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'img_id', 'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'
        ]
    ))


    test_pipeline = val_pipeline

    return train_pipeline, val_pipeline, test_pipeline