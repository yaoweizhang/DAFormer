#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class ACDCnightDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')

        super(ACDCnightDataset, self).__init__(
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                split=None,
                **kwargs)
