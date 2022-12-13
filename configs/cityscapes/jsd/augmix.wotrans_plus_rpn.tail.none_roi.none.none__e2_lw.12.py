_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_ai28.py',
    '../../_base_/datasets/cityscapes_detection.py',
    '../../_base_/schedules/ai28.py',
    '../../_base_/default_runtime.py'
]

'''
[NAMING] (updated 22.06.09)
  data pipeline:    [original, augmix(copy, wotrans)]
  loss:             [none, augmix, plus]
  rpn:              [rpn(none.none, jsd.none, ...)]
  roi:              [roi(none.none, jsd.none, ...)]
  parameters:       [e1, lw(e.g.,1e-4), wr(true, false)]
                     > e1: 1 epoch
                     > lw: lambda weight
                     > wr: weight reduce  
[OPTIONS] (updated 22.06.09)
  model
  * loss_cls/loss_bbox.additional_loss
    : [None, 'jsd', 'jsdy', 'jsdv1_1', 'jsdv2']
  * train_cfg.wandb.log.features_list 
    : [None, "rpn_head.rpn_cls", "neck.fpn_convs.0.conv", "neck.fpn_convs.1.conv", "neck.fpn_convs.2.conv", "neck.fpn_convs.3.conv"]
  * train_cfg.wandb.log.vars
    : ['log_vars'] 
'''

#############
### MODEL ###
#############
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0
            , additional_loss='rpn_tail', lambda_weight=12, wandb_name='rpn_cls'),
        loss_bbox=dict(type='L1LossPlus', loss_weight=1.0
                       , additional_loss="None", lambda_weight=0.0001, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0
                , additional_loss='None', lambda_weight=0.0001, wandb_name='roi_cls'),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0
                           , additional_loss="None", lambda_weight=0.0001, wandb_name='roi_bbox'))),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars']),
        )))

###############
### DATASET ###
###############
custom_imports = dict(imports=['mmdet.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### AugMix ###
    dict(type='AugMix', no_jsd=False, aug_list='wotrans', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'img3', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline)),
    test=dict(
        img_prefix='/ws/data/cityscapes-c/'+'leftImg8bit/val/'),)

################
### RUN TIME ###
################
runner = dict(
    type='EpochBasedRunner', max_epochs=2)  # actual epoch = 8 * 8 = 64
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[1])

###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]
# pipeline: [original, augmix(.copy, .wotrans')]
pipeline_list = ['AugMix', 'PixMix']
pipeline = next((item for item in train_pipeline if item['type'] in pipeline_list), None)
if pipeline is not None:
    str_pipeline = pipeline['type'].lower()
    str_pipeline += f".{pipeline['aug_list'].lower()}" if 'aug_list' in pipeline else ''
else:
    str_pipeline = 'original'
# loss type: [none, augmix, plus]
rpn_loss_cls = model['rpn_head']['loss_cls']
rpn_loss_bbox = model['rpn_head']['loss_bbox']
roi_loss_cls = model['roi_head']['bbox_head']['loss_cls']
roi_loss_bbox = model['roi_head']['bbox_head']['loss_bbox']
losses = [rpn_loss_cls, roi_loss_bbox, roi_loss_cls, roi_loss_bbox]
str_loss = 'none'
for i in range(len(losses)):
    if (losses[i]['type'] == 'CrossEntropyLossPlus') or (losses[i]['type'] == 'SmoothL1LossPlus'):
        str_loss = 'plus'
        break
    elif losses[i]['type'] == 'CrossEntropyLossAugMix':
        str_loss = 'augmix'
        break
# each loss type
if str_loss == 'plus':
    str_each_loss = "rpn"
    str_each_loss += f".{rpn_loss_cls['additional_loss'].lower()}"
    str_each_loss += f".{rpn_loss_bbox['additional_loss'].lower()}"
    str_each_loss += '_roi'
    str_each_loss += f".{roi_loss_cls['additional_loss'].lower()}"
    str_each_loss += f".{roi_loss_bbox['additional_loss'].lower()}"
elif str_loss == 'augmix':
    str_each_loss = "rpn"
    str_each_loss += f".{'jsd' if rpn_loss_cls['type'] == 'CrossEntropyLossAugMix' else 'none'}"
    str_each_loss += f".{'jsd' if (rpn_loss_bbox['type'] == 'L1LossAugMix') or (rpn_loss_bbox['type'] == 'SmoothL1LossAugMix') else 'none'}"
    str_each_loss += '_roi'
    str_each_loss += f".{'jsd' if roi_loss_cls['type'] == 'CrossEntropyLossAugMix' else 'none'}"
    str_each_loss += f".{'jsd' if (roi_loss_bbox['type'] == 'L1LossAugMix') or (roi_loss_bbox['type'] == 'SmoothL1LossAugMix') else 'none'}"
else:
    str_each_loss = "rpn.none.none_roi.none.none"
# parameters
str_parameters = '__'
str_parameters += 'e'+str(runner['max_epochs'])
# if 'lambda_weight' in rpn_loss_cls:
#     str_parameters += ('_lw.'+"{:.0e}".format(rpn_loss_cls['lambda_weight']))
# elif 'lambda_weight' in rpn_loss_bbox:
#     str_parameters += ('_lw'+"{:.0e}".format(rpn_loss_bbox['lambda_weight']))
# elif 'lambda_weight' in roi_loss_cls:
#     str_parameters += ('_lw'+"{:.0e}".format(roi_loss_cls['lambda_weight']))
# elif 'lambda_weight' in roi_loss_bbox:
#     str_parameters += ('_lw'+"{:.0e}".format(roi_loss_bbox['lambda_weight']))

print('++++++++++++++++++++')
print(f"{str_pipeline}_{str_loss}_{str_each_loss}{str_parameters}")
print('++++++++++++++++++++')

log_config = dict(interval=100,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLogger',
                           wandb_init_kwargs={'project': "AI28", 'entity': "kaist-url-ai28",
                                              'name': f"{str_pipeline}_{str_loss}_{str_each_loss}{str_parameters}",
                                              'config': {
                                                  # data pipeline
                                                  'data pipeline': f"{str_pipeline}",
                                                  # losses
                                                  'loss type(rpn_cls)': f"{rpn_loss_cls['type']}",
                                                  'loss type(rpn_bbox)': f"{rpn_loss_bbox['type']}",
                                                  'loss type(roi_cls)': f"{roi_loss_cls['type']}",
                                                  'loss type(roi_bbox)': f"{roi_loss_bbox['type']}",
                                                  # parameters
                                                  'epoch': runner['max_epochs'],
                                              }},
                           log_map_every_iter=False,
                           interval=500,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           num_eval_images=5),
                  ]
                  )

############
### LOAD ###
############
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
