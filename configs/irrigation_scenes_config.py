import os

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True

custom_imports = dict(imports=["geospatial_fm"])


### Configs
# Data
# TO BE DEFINED BY USER: Data root to firescar downloaded dataset
data_root = "data/irrigation_scenes/"

dataset_type = "SpatioTemporalDataset"
num_classes = 1
num_frames = int(os.getenv("NUM_FRAMES", 3))
img_size = int(os.getenv("IMG_SIZE", 224))
num_workers = int(os.getenv("DATA_LOADER_NUM_WORKERS", 2))
samples_per_gpu = 1
CLASSES = (0, 1)

img_norm_cfg = dict(
    means=[0.166, 0.166, 0.166, 0.166, 0.166, 0.166],
    stds=[0.114, 0.114, 0.114, 0.114, 0.114, 0.114],
)
# Sentinel-2 Bands 2,3,4,8A,11,12 (Blue, Green, Red, NIR_Narrow, SWIR1, SWIR2)
bands = [0, 1, 2, 3, 4, 5]

tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

img_suffix = ".tif"
seg_map_suffix = ".tif"


# ignore_index = -1
# image_nodata = -9999
# image_nodata_replace = 0
image_to_float32 = True

# Model
# TO BE DEFINED BY USER: path to pretrained backbone weights
pretrained_weights_path = "pretrain_ckpts/Prithvi_100M.pt"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1

# TRAINING
# epochs=50
# eval_epoch_interval = 5

# TO BE DEFINED BY USER: Save directory
experiment = "test_1"
project_dir = "finetune_weights/irrigation_scenes"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

gpu_ids = [0]

splits = {
    "train": data_root + "training_chips/training_data.txt",
    "val": data_root + "validation_chips/validation_data.txt",
    "test": data_root + "validation_chips/validation_data.txt",
}

# Pipelines
train_pipeline = [
    dict(
        type="LoadSpatioTemporalImagesFromFile",
        to_float32=image_to_float32,
        channels_last=True,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=255,
        nodata_replace=2,
    ),
    dict(type="RandomFlip", prob=0.5),  # flip on axis 1, assume channel last NHWC
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(
        type="TorchPermute",
        keys=["img"],
        order=(0, 3, 1, 2),  # channel last to channels first NCHW
    ),
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

val_pipeline = [
    dict(
        type="LoadSpatioTemporalImagesFromFile",
        to_float32=image_to_float32,
        channels_last=True,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=255,
        nodata_replace=2,
    ),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    dict(
        type="TorchPermute",
        keys=["img"],
        order=(0, 3, 1, 2),  # channel last to channels first NCHW
    ),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(
        type="Collect",
        keys=["img", "gt_semantic_seg"],
        meta_keys=[
            "img_info",
            "ann_info",
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
            "gt_semantic_seg",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadSpatioTemporalImagesFromFile", to_float32=image_to_float32),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up={"2": 1, "3": 2},
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

CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="month1",
        ann_dir="masks",
        pipeline=train_pipeline,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        # split=splits["train"],
    ),
    val=dict(
        type=dataset_type,
        # CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="month1",
        ann_dir="masks",
        pipeline=val_pipeline,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        # split=splits["val"],
    ),
    test=dict(
        type=dataset_type,
        # CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="month1",
        ann_dir="masks",
        pipeline=test_pipeline,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        # split=splits["test"],
    ),
)
# gt_seg_map_loader_cfg=dict(nodata=-1, nodata_replace=2)))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type="Adam", lr=1.5e-5, betas=(0.9, 0.999), weight_decay=0.05)
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
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook", by_epoch=False),
    ],
)

checkpoint_config = dict(by_epoch=True, interval=10, out_dir=save_path)

evaluation = dict(
    interval=1180, metric="mIoU", pre_eval=True, save_best="mIoU", by_epoch=False
)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)

optimizer_config = dict(grad_clip=None)

runner = dict(type="IterBasedRunner", max_iters=10000)
workflow = [("train", 1)]

norm_cfg = dict(type="BN", requires_grad=True)

loss_weights_multi = [
    1.5652886,
    0.46067129,
    0.59387921,
    0.48431193,
    0.65555127,
    0.73865282,
    0.77616475,
    3.46336277,
    1.01650963,
    1.87640752,
    1.52960976,
    1.49788817,
    57.55048277,
    1.97697006,
    2.34793961,
    0.83456613,
]

# loss_func = dict(type='DiceLoss', use_sigmoid=False, loss_weight=1, class_weight=loss_weights_multi)
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)


output_embed_dim = embed_dim * num_frames

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
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=len(loss_weights_multi),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(loss_weights_multi),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
