# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NeurocleSuncheonDataset(BaseSegDataset):
    """NeurocleSuncheonDataset.

    """
    METAINFO = dict(
        classes=(
            'Hemorrhage'),
        palette=[[0, 192, 64]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
