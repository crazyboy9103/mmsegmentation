# optimizer
optimizer = dict(type='AdamW', lr=0.0006, weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=24,
        by_epoch=True)
]
# training schedule for 10k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
