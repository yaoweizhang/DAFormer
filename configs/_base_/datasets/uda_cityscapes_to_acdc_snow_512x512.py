# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
#[128.84636057 126.44753499 128.95696975]
#[58.7353162  59.40295156 62.33273263]

img_norm_cfg = dict(
    mean=[128.85, 126.45, 128.96], std=[58.74, 59.40, 62.33], to_rgb=True)
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
acdc_snow_train_pipeline = [
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
            type='ACDCsnowDataset',
            data_root='data/ACDC/',
            img_dir='train_snow',
            ann_dir='gt_train_snow',
            pipeline=acdc_snow_train_pipeline)
        ),
    val=dict(
        type='ACDCsnowDataset',
        data_root='data/ACDC/',
        img_dir='val_snow',
        ann_dir='gt_val_snow',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCsnowDataset',
        data_root='data/ACDC/',
        img_dir='test_snow',
        #ann_dir='gt_val_snow',
        pipeline=test_pipeline))
