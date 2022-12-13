# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/ws/data/cityscapes/'
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### Insert PixMix ###
    dict(type='PixMix'
         , beta=3   # Severity of mixing
         , k=4  # Mixing iterations
         , aug_list='all_ops'   # Turn on all augmentation operations (+brightness,contrast,color,sharpness).
         , aug_severity=3
         , mixing_set='/ws/data/fractals_and_fvis'
         , use_300k=False
         ),   # Severity of base augmentation operators
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'img3', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline))
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root +
    #     'annotations/instancesonly_filtered_gtFine_test.json',
    #     img_prefix=data_root + 'leftImg8bit/test/',
    #     pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
