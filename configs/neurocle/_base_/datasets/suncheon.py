_base_ = ["./customdata_base.py"]

dataset_type = 'NeurocleSuncheonDataset'
data_root = 'data/transformer/seg/suncheon'

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [0.7945, 1.2055]
num_classes = 2

train_dataloader = dict(
    dataset=dict(
        type=dataset_type, 
        data_root=data_root
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type, 
        data_root=data_root
    )
)
test_dataloader = val_dataloader