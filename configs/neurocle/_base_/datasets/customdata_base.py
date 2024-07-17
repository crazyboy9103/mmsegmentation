_base_ = ["./coco-stuff164k.py"]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale={{_base_.crop_size}}, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale={{_base_.crop_size}}, keep_ratio=False), #False for images with various sizes
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transforms
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'
        ),
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader