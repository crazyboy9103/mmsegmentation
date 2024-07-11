_base_ = ["./customdata_base.py"]

dataset_type = 'NeurocleSeverstalDataset'
data_root = 'data/transformer/seg/severstal'

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [0.7604, 0.9782, 1.112, 0.9689, 1.1806]
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