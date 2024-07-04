# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NeurocleSeverstalDataset(BaseSegDataset):
    """NeurocleSeverstalDataset.

    """
    METAINFO = dict(
        classes=(
            'BG', '90', '92', '91', '85'),
        palette=[[165, 42, 42], [0, 192, 64], [165, 42, 42], [0, 192, 0], [196, 196, 196]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
