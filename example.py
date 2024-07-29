checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
class_weight = [
    0.8283,
    1.0177,
    0.9453,
    0.956,
    0.9761,
    0.9343,
    0.9338,
    0.9437,
    0.9876,
    1.0669,
    1.0367,
    1.0367,
    1.0662,
    0.9724,
    1.0161,
    0.9464,
    0.9619,
    0.9721,
    1.0062,
    0.9875,
    0.9562,
    1.008,
    0.9765,
    0.9763,
    1.0495,
    0.9824,
    1.0421,
    1.1265,
    0.9788,
    1.1199,
    1.1201,
    1.1096,
    1.1814,
    1.0686,
    1.1762,
    1.1569,
    1.0723,
    1.033,
    1.078,
    1.0169,
    1.052,
    0.9914,
    1.1271,
    1.0995,
    1.1369,
    0.9457,
    0.9973,
    1.0398,
    0.9822,
    1.0266,
    1.0094,
    1.065,
    1.03,
    0.9357,
    1.0095,
    0.9711,
    0.9428,
    0.9482,
    0.9955,
    0.9204,
    0.871,
    0.9784,
    0.9748,
    0.9714,
    1.1442,
    1.0822,
    1.0235,
    1.0461,
    1.0489,
    0.9713,
    1.2437,
    1.0104,
    0.969,
    1.0106,
    1.0281,
    1.0125,
    1.1114,
    0.9781,
    1.2982,
    1.1692,
    0.9957,
    1.036,
    0.9999,
    1.0324,
    0.8728,
    0.9463,
    0.9326,
    0.9612,
    0.9967,
    0.9573,
    0.9251,
    1.125,
    1.0442,
    0.971,
    0.8823,
    0.9788,
    1.0473,
    0.9563,
    0.9811,
    0.9051,
    0.941,
    0.9245,
    1.045,
    0.9389,
    1.0158,
    0.9419,
    0.9464,
    1.0042,
    0.951,
    0.9558,
    1.0297,
    0.9269,
    0.8582,
    0.9867,
    0.9366,
    0.9961,
    0.9625,
    0.9863,
    1.0242,
    1.1355,
    0.9277,
    0.9727,
    1.1586,
    0.954,
    1.0547,
    1.0702,
    1.0081,
    0.9647,
    0.885,
    1.0819,
    0.9322,
    0.9696,
    0.9973,
    0.8943,
    1.0557,
    0.9844,
    0.9509,
    0.8815,
    0.9788,
    1.0217,
    1.0144,
    1.0857,
    0.9208,
    0.883,
    0.985,
    0.8426,
    0.9901,
    0.889,
    1.0822,
    1.0373,
    1.0223,
    1.007,
    1.0124,
    0.9142,
    1.0545,
    0.9492,
    1.0593,
    0.8456,
    1.0003,
    0.9503,
    0.8563,
    0.8861,
    0.9871,
    0.9976,
    0.9264,
    0.9428,
    0.9617,
    1.1853,
    1.0115,
    0.9208,
    0.9907,
]
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/transformer/seg/coco-stuff/'
dataset_type = 'COCOStuffDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best=[
            'mIoU',
        ],
        save_last=False,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=171,
        type='SegformerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 171
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(seed=304)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        data_root='data/transformer/seg/coco-stuff/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='COCOStuffDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1024,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train2017', seg_map_path='annotations/train2017'),
        data_root='data/transformer/seg/coco-stuff/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='COCOStuffDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        data_root='data/transformer/seg/coco-stuff/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1024,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='COCOStuffDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            project='neurocle', tags=[
                'seg',
                'segformer',
                'coco-stuff',
            ]),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                project='neurocle', tags=[
                    'seg',
                    'segformer',
                    'coco-stuff',
                ]),
            type='WandbVisBackend'),
    ])
