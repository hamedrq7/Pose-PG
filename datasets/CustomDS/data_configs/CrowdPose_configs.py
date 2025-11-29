CrowdPose_data_root = 'datasets/CrowdPose'

CrowdPose_dataset_info = dict(
    dataset_name='crowdpose',
    paper_info=dict(
        author='Li, Jiefeng and Wang, Can and Zhu, Hao and '
        'Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu',
        title='CrowdPose: Efficient Crowded Scenes Pose Estimation '
        'and A New Benchmark',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2019',
        homepage='https://github.com/Jeff-sjtu/CrowdPose',
    ),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(
            name='top_head', id=12, color=[255, 128, 0], type='upper',
            swap=''),
        13:
        dict(name='neck', id=13, color=[0, 255, 0], type='upper', swap='')
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
        dict(link=('top_head', 'neck'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('right_shoulder', 'neck'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('left_shoulder', 'neck'), id=14, color=[51, 153, 255])
    },
    joint_weights=[
        0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5
    ],
    sigmas=[
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.079, 0.079
    ])



CrowdPose_channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


def get_data_cfg(image_resolution): 
    W, H = image_resolution[1], image_resolution[0]
    heatmap_size = [W//4, H//4]
    if (W==192 and H==256):
        heatmap_sigma = 2.0 
    elif (W==288 and H==384):
        heatmap_sigma = 3.0

    CrowdPose_data_cfg = dict(
        image_size=[W, H],
        heatmap_size=heatmap_size,
        heatmap_sigma = heatmap_sigma, 
        num_output_channels=CrowdPose_channel_cfg['num_output_channels'],
        num_joints=CrowdPose_channel_cfg['dataset_joints'],
        dataset_channel=CrowdPose_channel_cfg['dataset_channel'],
        inference_channel=CrowdPose_channel_cfg['inference_channel'],
        crowd_matching=False,
        soft_nms=False,
        nms_thr=1.0,
        oks_thr=0.9,
        vis_thr=0.2,
        use_gt_bbox=False,
        det_bbox_thr=0.0,
        bbox_file=f'{CrowdPose_data_root}/annotations/'
        'det_for_crowd_test_0.1_0.5.json',
    )

    return CrowdPose_data_cfg

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

def get_pipelines(image_resolution, model_name, no_normalization: bool = False):
    data_cfg = get_data_cfg(image_resolution)

    # if model_name == "poseresnet" or model_name == 'hrnet':
    #     udp = False
    # elif model_name == "vitpose_small":
    #     udp = True

    if model_name == 'vitpose_small':
        udp = True
    else:
        udp = False
    

    train_pipeline = [LoadImageFromFile()]
    train_pipeline.append(TopDownRandomFlip(flip_prob=0.5))
    train_pipeline.append(TopDownHalfBodyTransform(num_joints_half_body=6, prob_half_body=0.3))
    train_pipeline.append(TopDownGetRandomScaleRotation(rot_factor=40., scale_factor=0.5))
    train_pipeline.append(TopDownAffine(use_udp=udp))
    train_pipeline.append(ToTensor()) 
    if not no_normalization:
        train_pipeline.append(NormalizeTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        )
    train_pipeline.append(TopDownGenerateTarget(sigma=data_cfg['heatmap_sigma'], encoding='UDP' if udp else 'MSRA'))
    train_pipeline.append(
        Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'img_id', 'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'
        ]
    ))

    val_pipeline = [LoadImageFromFile()]
    val_pipeline.append(TopDownAffine(use_udp=udp))
    val_pipeline.append(ToTensor())
    if not no_normalization:
        val_pipeline.append(NormalizeTensor(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ))
    val_pipeline.append(TopDownGenerateTarget(sigma=data_cfg['heatmap_sigma'], encoding='UDP' if udp else 'MSRA')) 
    val_pipeline.append(Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'img_id', 'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'
        ]))

    test_pipeline = val_pipeline
    return train_pipeline, val_pipeline, test_pipeline
