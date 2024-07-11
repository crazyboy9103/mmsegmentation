_base_ = ["./customdata_base.py"]

dataset_type = 'NeurocleCosmeticDataset'
data_root = 'data/transformer/seg/cosmetic'

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [0.8455, 1.1545]
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