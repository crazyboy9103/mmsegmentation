_base_ = ["./custom_base.py"]

dataset_type = 'NeurocleCosmeticDataset'
data_root = '/datasets/seg/cosmetic'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,))
test_dataloader = val_dataloader