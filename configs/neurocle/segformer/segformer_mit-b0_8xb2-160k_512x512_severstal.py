_base_ = [
    './custom_base.py', '../_base_/datasets/cosmetic.py',
]

vis_backends = [
    dict(type='LocalVisBackend'), 
    dict(type='WandbVisBackend', init_kwargs=dict(project='neurocle', tags=['seg', 'segformer', 'cosmetic']))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    decode_head=dict(num_classes=5))