_base_ = ["./custom_base.py"]

dataset_type = 'NeurocleSuncheonDataset'
data_root = 'data/transformer/seg/suncheon'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,))
test_dataloader = val_dataloader