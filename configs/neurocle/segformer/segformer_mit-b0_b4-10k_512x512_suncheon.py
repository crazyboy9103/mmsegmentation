_base_ = [
    '../_base_/models/segformer_mit-b0.py', 
    '../_base_/datasets/suncheon.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_10k.py'
]

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['seg', 'segformer', 'suncheon']))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    decode_head=dict(
        num_classes={{_base_.num_classes}}
    ),
    data_preprocessor=dict(size={{_base_.crop_size}})
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
)

val_dataloader = dict(
    batch_size=32, 
    num_workers=4
)