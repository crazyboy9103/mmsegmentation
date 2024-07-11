_base_ = ["./coco-stuff164k.py"]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=False), #False for images with various sizes
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'
        ),
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