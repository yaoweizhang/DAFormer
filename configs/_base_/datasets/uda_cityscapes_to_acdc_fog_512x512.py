# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
#[128.66016052 126.15703683 128.00904034]
#[60.66112264 61.96867872 65.8003836 ]
img_norm_cfg = dict(
    mean=[128.66, 126.16, 128.01], std=[60.66, 61.97, 65.8], to_rgb=True)
crop_size = (512, 512)
#gta_train_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='LoadAnnotations'),
#    dict(type='Resize', img_scale=(1280, 720)),
#    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#    dict(type='RandomFlip', prob=0.5),
#    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
#    dict(type='Normalize', **img_norm_cfg),
#    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#    dict(type='DefaultFormatBundle'),
#    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
#]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
acdc_fog_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='ACDCfogDataset',
            data_root='data/ACDC/',
            img_dir='rgb_anon/fog/train',
            ann_dir='gt/fog/train',
            pipeline=acdc_fog_train_pipeline)
        ),
    val=dict(
        type='ACDCfogDataset',
        data_root='data/ACDC/',
        img_dir='rgb_anon/fog/val',
        ann_dir='gt/fog/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCfogDataset',
        data_root='data/ACDC/',
        img_dir='rgb_anon/fog/test',
        #ann_dir='gt_val_fog',
        pipeline=test_pipeline))
