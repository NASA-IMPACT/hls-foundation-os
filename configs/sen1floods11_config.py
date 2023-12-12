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
# TO BE DEFINED BY USER: Data root to sen1floods11 downloaded dataset
data_root = "<path to root directory of sen1floods11 dataset>"

dataset_type = "GeospatialDataset"
num_classes = 2
num_frames = 1
img_size = 224
num_workers = 2
samples_per_gpu = 4
CLASSES = (0, 1)

img_norm_cfg = dict(
    means=[0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526, 0.12046503],
    stds=[0.04036231, 0.04186983, 0.05267646, 0.0822221, 0.06834774, 0.05294205],
)

bands = [1, 2, 3, 8, 11, 12]
tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

img_dir = data_root + "v1.1/data/flood_events/HandLabeled/S2Hand"
ann_dir = data_root + "v1.1/data/flood_events/HandLabeled/LabelHand"
img_suffix = f"_S2Hand.tif"
seg_map_suffix = f"_LabelHand.tif"

splits = {
    "train": "data_splits/sen1floods11/train_split.txt",
    "val": "data_splits/sen1floods11/val_split.txt",
    "test": "data_splits/sen1floods11/test_split.txt",
}
splits = {k: os.path.abspath(v) for (k, v) in splits.items()}

ignore_index = 2
label_nodata = -1
image_nodata = -9999
image_nodata_replace = 0
constant = 0.0001

# Model
# TO BE DEFINED BY USER: path to pretrained backbone weights
pretrained_weights_path = "<path to pretrained weights>"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1

# TRAINING
epochs = 100
eval_epoch_interval = 5

# TO BE DEFINED BY USER: Save directory
experiment = "<experiment name>"
project_dir = "<project dir>"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

# Pipelines
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=False,
        nodata=image_nodata,
        nodata_replace=image_nodata_replace,
    ),
    dict(
        type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=label_nodata,
        nodata_replace=ignore_index,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ConstantMultiply", constant=constant),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
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
        to_float32=False,
        nodata=image_nodata,
        nodata_replace=image_nodata_replace,
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ConstantMultiply", constant=constant),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
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

# Dataset
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=ignore_index,
        split=splits["train"],
    ),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
        split=splits["val"],
        gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
    ),
    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
        split=splits["test"],
        gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index),
    ),
)

# Training
optimizer = dict(
    type="AdamW",
    lr=1.5e-5,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)
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
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(type="TensorboardLoggerHook", by_epoch=True),
    ],
)

checkpoint_config = dict(by_epoch=True, interval=10, out_dir=save_path)

evaluation = dict(
    interval=eval_epoch_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True,
)

runner = dict(type="EpochBasedRunner", max_epochs=epochs)

workflow = [("train", 1), ("val", 1)]

norm_cfg = dict(type="BN", requires_grad=True)

ce_weights = [0.3, 0.7]

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
        embed_dim=num_frames * embed_dim,
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
        ignore_index=ignore_index,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            avg_non_ignore=True,
        ),
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        in_channels=embed_dim,
        ignore_index=ignore_index,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
            avg_non_ignore=True,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
