dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=['geospatial_fm'])
num_frames = 3
img_size = 224
num_workers = 2
pretrained_weights_path = '/home/ubuntu/hls-loss-weights/Prithvi_100M.pt'
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
epochs = 80
eval_epoch_interval = 2
experiment = 'multiclass_exp_newSplit'
work_dir = '/home/ubuntu/clark_gfm_eval/multiclass_exp_newSplit'
save_path = '/home/ubuntu/clark_gfm_eval/multiclass_exp_newSplit'
gpu_ids = range(0, 1)
dataset_type = 'GeospatialDataset'
data_root = '/home/ubuntu/hls_cdl_reclassed/'
img_norm_cfg = dict(
    means=[
        494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,
        1739.579917, 494.905781, 815.239594, 924.335066, 2968.881459,
        2634.621962, 1739.579917, 494.905781, 815.239594, 924.335066,
        2968.881459, 2634.621962, 1739.579917
    ],
    stds=[
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808,
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808,
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808
    ])
splits = dict(
    train=
    '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/training_data.txt',
    val=
    '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/validation_data.txt',
    test=
    '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/validation_data.txt'
)
bands = [0, 1, 2, 3, 4, 5]
tile_size = 224
orig_nsize = 512
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(
        type='TorchNormalize',
        means=[
            494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,
            1739.579917, 494.905781, 815.239594, 924.335066, 2968.881459,
            2634.621962, 1739.579917, 494.905781, 815.239594, 924.335066,
            2968.881459, 2634.621962, 1739.579917
        ],
        stds=[
            284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
            921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
            951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
            896.601013, 951.900334, 921.407808
        ]),
    dict(type='TorchRandomCrop', crop_size=(224, 224)),
    dict(type='Reshape', keys=['img'], new_shape=(6, 3, 224, 224)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, 224, 224)),
    dict(
        type='CastTensor',
        keys=['gt_semantic_seg'],
        new_type='torch.LongTensor'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(
        type='TorchNormalize',
        means=[
            494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,
            1739.579917, 494.905781, 815.239594, 924.335066, 2968.881459,
            2634.621962, 1739.579917, 494.905781, 815.239594, 924.335066,
            2968.881459, 2634.621962, 1739.579917
        ],
        stds=[
            284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
            921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
            951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
            896.601013, 951.900334, 921.407808
        ]),
    dict(type='TorchRandomCrop', crop_size=(224, 224)),
    dict(type='Reshape', keys=['img'], new_shape=(6, 3, 224, 224)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, 224, 224)),
    dict(
        type='CastTensor',
        keys=['gt_semantic_seg'],
        new_type='torch.LongTensor'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'img_info', 'ann_info', 'seg_fields', 'img_prefix', 'seg_prefix',
            'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape',
            'pad_shape', 'scale_factor', 'img_norm_cfg', 'gt_semantic_seg'
        ])
]
test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='ToTensor', keys=['img']),
    dict(
        type='TorchNormalize',
        means=[
            494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,
            1739.579917, 494.905781, 815.239594, 924.335066, 2968.881459,
            2634.621962, 1739.579917, 494.905781, 815.239594, 924.335066,
            2968.881459, 2634.621962, 1739.579917
        ],
        stds=[
            284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
            921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
            951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
            896.601013, 951.900334, 921.407808
        ]),
    dict(
        type='Reshape',
        keys=['img'],
        new_shape=(6, 3, -1, -1),
        look_up=dict({
            '2': 1,
            '3': 2
        })),
    dict(type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
    dict(
        type='CollectTestList',
        keys=['img'],
        meta_keys=[
            'img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename',
            'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape',
            'scale_factor', 'img_norm_cfg'
        ])
]
CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='GeospatialDataset',
        CLASSES=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        reduce_zero_label=True,
        data_root='/home/ubuntu/hls_cdl_reclassed/',
        img_dir='/home/ubuntu/hls_cdl_reclassed/training_chips',
        ann_dir='/home/ubuntu/hls_cdl_reclassed/training_chips',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
            dict(
                type='TorchNormalize',
                means=[
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917, 494.905781, 815.239594,
                    924.335066, 2968.881459, 2634.621962, 1739.579917,
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917
                ],
                stds=[
                    284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
                    921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
                    951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
                    896.601013, 951.900334, 921.407808
                ]),
            dict(type='TorchRandomCrop', crop_size=(224, 224)),
            dict(type='Reshape', keys=['img'], new_shape=(6, 3, 224, 224)),
            dict(
                type='Reshape',
                keys=['gt_semantic_seg'],
                new_shape=(1, 224, 224)),
            dict(
                type='CastTensor',
                keys=['gt_semantic_seg'],
                new_type='torch.LongTensor'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=
        '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/training_data.txt'
    ),
    val=dict(
        type='GeospatialDataset',
        CLASSES=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        reduce_zero_label=True,
        data_root='/home/ubuntu/hls_cdl_reclassed/',
        img_dir='/home/ubuntu/hls_cdl_reclassed/validation_chips',
        ann_dir='/home/ubuntu/hls_cdl_reclassed/validation_chips',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(
                type='TorchNormalize',
                means=[
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917, 494.905781, 815.239594,
                    924.335066, 2968.881459, 2634.621962, 1739.579917,
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917
                ],
                stds=[
                    284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
                    921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
                    951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
                    896.601013, 951.900334, 921.407808
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 3, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=
        '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/validation_data.txt'
    ),
    test=dict(
        type='GeospatialDataset',
        CLASSES=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        reduce_zero_label=True,
        data_root='/home/ubuntu/hls_cdl_reclassed/',
        img_dir='/home/ubuntu/hls_cdl_reclassed/validation_chips',
        ann_dir='/home/ubuntu/hls_cdl_reclassed/validation_chips',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(
                type='TorchNormalize',
                means=[
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917, 494.905781, 815.239594,
                    924.335066, 2968.881459, 2634.621962, 1739.579917,
                    494.905781, 815.239594, 924.335066, 2968.881459,
                    2634.621962, 1739.579917
                ],
                stds=[
                    284.925432, 357.84876, 575.566823, 896.601013, 951.900334,
                    921.407808, 284.925432, 357.84876, 575.566823, 896.601013,
                    951.900334, 921.407808, 284.925432, 357.84876, 575.566823,
                    896.601013, 951.900334, 921.407808
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 3, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=
        '/home/ubuntu/hls-foundation-os/fine-tuning-examples/data_splits/crop_classification/validation_data.txt'
    ))
optimizer = dict(
    type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(
    by_epoch=True,
    interval=10,
    out_dir='/home/ubuntu/clark_gfm_eval/multiclass_exp_newSplit')
evaluation = dict(interval=2, metric='mIoU', pre_eval=True, save_best='mIoU')
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type='EpochBasedRunner', max_epochs=80)
workflow = [('train', 1), ('val', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
loss_weights_multi = [
    0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
    1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=[
        0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
        1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
    ],
    avg_non_ignore=True)
output_embed_dim = 2304
model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained='/home/ubuntu/hls-loss-weights/Prithvi_100M.pt',
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=6,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=2304,
        output_embed_dim=2304,
        drop_cls_token=True,
        Hp=14,
        Wp=14),
    decode_head=dict(
        num_classes=13,
        in_channels=2304,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[
                0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186,
                3.249462, 1.542289, 2.175141, 2.272419, 3.062762, 3.626097,
                1.198702
            ],
            avg_non_ignore=True)),
    auxiliary_head=dict(
        num_classes=13,
        in_channels=2304,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[
                0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186,
                3.249462, 1.542289, 2.175141, 2.272419, 3.062762, 3.626097,
                1.198702
            ],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(112, 112), crop_size=(224, 224)))
auto_resume = False
