_base_ = [
    '../_base_/datasets/coco-stuff164k.py',
    '../_base_/models/pidnet-s.py',
    '../_base_/schedules/schedule_160k.py',
    '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['seg', 'pidnet', 'coco-stuff']))
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

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
)

val_dataloader = dict(
    batch_size=8, 
    num_workers=4
)