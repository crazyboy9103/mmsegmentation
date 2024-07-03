# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NeurocleCosmeticDataset(BaseSegDataset):
    """NeurocleCosmeticDataset.

    """
    METAINFO = dict(
        classes=(
            'BG', 'NG'),
        palette=[[165, 42, 42], [0, 192, 64]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
