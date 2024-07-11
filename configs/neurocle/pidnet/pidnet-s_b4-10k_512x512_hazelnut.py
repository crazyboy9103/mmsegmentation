_base_ = [
    '../_base_/datasets/hazelnut.py',
    '../_base_/models/pidnet-s.py',
    '../_base_/schedules/schedule_10k.py',
    '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['seg', 'pidnet', 'hazelnut']))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    decode_head=dict(
        num_classes={{_base_.num_classes}},
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight={{_base_.class_weight}}, 
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight={{_base_.class_weight}}, 
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight={{_base_.class_weight}}, 
                loss_weight=1.0)
        ]),
    data_preprocessor=dict(size={{_base_.crop_size}})
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

