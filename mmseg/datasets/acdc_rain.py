#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class ACDCrainDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self,
                img_suffix='_rgb_anon.png',
                seg_map_suffix='_gt_labelTrainIds.png',
                **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')

        super(ACDCrainDataset, self).__init__(
                img_suffix=img_suffix,
                seg_map_suffix=seg_map_suffix,
                split=None,
                **kwargs)
