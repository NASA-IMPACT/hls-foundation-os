import os

# base options
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True

custom_imports = dict(imports=["geospatial_fm"])


### Configs
# Data
# TO BE DEFINED BY USER: Data root to sen1floods11 downloaded dataset
data_root = "<path to firescars root>"

dataset_type = "GeospatialDataset"
num_classes=2
num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4
CLASSES=(0,1)

img_norm_cfg = dict(
    means=[0.033349706741586264, 0.05701185520536176, 0.05889748132001316, 0.2323245113436119,
           0.1972854853760658, 0.11944914225186566],
    stds=[0.02269135568823774, 0.026807560223070237, 0.04004109844362779, 0.07791732423672691,
          0.08708738838140137, 0.07241979477437814])  ## change the mean and std of all the bands

bands = [0, 1, 2, 3, 4, 5]

tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

img_suffix = "_merged.tif"
seg_map_suffix = ".mask.tif"


ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True

# Model
# TO BE DEFINED BY USER: path to pretrained backbone weights
pretrained_weights_path = "<path to pretrained weights>"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1

# TRAINING
epochs=50
eval_epoch_interval = 5

# TO BE DEFINED BY USER: Save directory
experiment = "<experiment name>"
project_dir = "<project directory>"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

# Pipelines
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]


test_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up={'2': 1, '3': 2}
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

# Dataset
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir='training',
        ann_dir='training',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir='validation',
        ann_dir='validation',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        CLASSES,CLASSES
        data_root=data_root,
        img_dir='validation',
        ann_dir='validation',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index
    ),
)

# Training
optimizer = dict(type="Adam", lr=1.5e-5, betas=(0.9, 0.999),  weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

checkpoint_config = dict(
    by_epoch=True, interval=10, out_dir=save_path, 
)

evaluation = dict(
    interval=1180, metric="mIoU", pre_eval=True, save_best="mIoU", by_epoch=False
)

# runner = dict(type="EpochBasedRunner", max_epochs=epochs)
runner = dict(type='IterBasedRunner', max_iters=6300)
workflow = [("train", 1)]

norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=num_frames*embed_dim,
        output_embed_dim=embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1, ignore_index=ignore_index),
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='DiceLoss', use_sigmoid=False, loss_weight=1, ignore_index=ignore_index),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", stride=(int(tile_size/2), int(tile_size/2)), crop_size=(tile_size, tile_size)),
)
