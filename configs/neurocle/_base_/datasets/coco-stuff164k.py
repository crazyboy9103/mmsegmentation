# dataset settings
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

crop_size = (512, 512)
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
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train2017', seg_map_path='annotations/train2017'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
