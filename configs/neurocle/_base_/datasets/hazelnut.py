_base_ = ["./customdata_base.py"]

dataset_type = 'NeurocleHazelnutDataset'
data_root = 'data/transformer/seg/hazelnut'

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [0.6865, 0.9665, 1.1132, 1.0663, 1.1675]
num_classes = 5

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