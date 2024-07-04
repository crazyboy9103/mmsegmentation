_base_ = [
    '../_base_/datasets/coco-stuff164k.py',
    '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['pidnet', 'coco-stuff']))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

dataset_type = 'COCOStuffDataset'
data_root = 'data/transformer/seg/coco-stuff/'

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [0.8283, 1.0177, 0.9453, 0.956, 0.9761, 0.9343, 0.9338, 0.9437, 0.9876, 1.0669, 
                1.0367, 1.0367, 1.0662, 0.9724,1.0161, 0.9464, 0.9619, 0.9721, 1.0062, 0.9875,
                0.9562, 1.008, 0.9765, 0.9763, 1.0495, 0.9824, 1.0421, 1.1265, 0.9788, 1.1199,
                1.1201, 1.1096, 1.1814, 1.0686, 1.1762, 1.1569, 1.0723, 1.033, 1.078, 1.0169, 
                1.052, 0.9914, 1.1271, 1.0995, 1.1369, 0.9457, 0.9973, 1.0398, 0.9822, 1.0266, 
                1.0094, 1.065, 1.03, 0.9357, 1.0095, 0.9711, 0.9428, 0.9482, 0.9955, 0.9204,
                0.871, 0.9784, 0.9748, 0.9714, 1.1442, 1.0822, 1.0235, 1.0461, 1.0489, 0.9713,
                1.2437, 1.0104, 0.969, 1.0106, 1.0281, 1.0125, 1.1114, 0.9781, 1.2982, 1.1692,
                0.9957, 1.036, 0.9999, 1.0324, 0.8728, 0.9463, 0.9326, 0.9612, 0.9967, 0.9573,
                0.9251, 1.125, 1.0442, 0.971, 0.8823, 0.9788, 1.0473, 0.9563, 0.9811, 0.9051,
                0.941, 0.9245, 1.045, 0.9389, 1.0158, 0.9419, 0.9464, 1.0042, 0.951, 0.9558, 
                1.0297, 0.9269, 0.8582, 0.9867, 0.9366, 0.9961, 0.9625, 0.9863, 1.0242, 1.1355,
                0.9277, 0.9727, 1.1586, 0.954, 1.0547, 1.0702, 1.0081, 0.9647, 0.885, 1.0819,
                0.9322, 0.9696, 0.9973, 0.8943, 1.0557, 0.9844, 0.9509, 0.8815, 0.9788, 1.0217,
                1.0144, 1.0857, 0.9208, 0.883, 0.985, 0.8426, 0.9901, 0.889, 1.0822, 1.0373,
                1.0223, 1.007, 1.0124, 0.9142, 1.0545, 0.9492, 1.0593, 0.8456, 1.0003, 0.9503,
                0.8563, 0.8861, 0.9871, 0.9976, 0.9264, 0.9428, 0.9617, 1.1853, 1.0115, 0.9208, 
                0.9907]
num_classes = 171 
assert len(class_weight) == num_classes

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8)
test_dataloader = val_dataloader

iters = 120000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
