import os
import sys

# base options
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True


project_root = '/u/fraccaro/Projects/general_libraries/hls-foundation-os/'
sys.path.append(project_root)

custom_imports = dict(imports=["sen1floods11.geospatial_fm"])

### Configs
# Data
dataset_type = "Sen1Floods11"
# TO BE DEFINED BY USER: Data root to HandLabeled folder ([...]/sen1floods11/v1.1/data/flood_events/HandLabeled)
data_root = "/dccstor/geofm-finetuning/carlosgomes/sen1floods11/"
num_frames = 1
img_size = 224
num_workers = 2

img_norm_cfg = dict(means=[0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526,0.12046503],
                    stds=[0.04036231, 0.04186983, 0.05267646, 0.0822221 , 0.06834774, 0.05294205])

bands = [1, 2, 3, 8, 11, 12]
tile_size = img_size
orig_nsize = 512
crop_size = (tile_size, tile_size)

img_dir = data_root + "v1.1/data/flood_events/HandLabeled/S2Hand"
ann_dir = data_root + "v1.1/data/flood_events/HandLabeled/LabelHand"
img_suffix = f"_S2Hand.tif"
seg_map_suffix = f"_LabelHand.tif"


splits = {
    "train": project_root + "sen1floods11/data_splits/train_split.txt",
    "val": project_root + "sen1floods11/data_splits/val_split.txt",
    "test": project_root + "sen1floods11/data_splits/test_split.txt",
}

splits = {k: os.path.abspath(v) for (k, v) in splits.items()}

ignore_index = 2
label_nodata = -1
image_nodata = -9999
image_nodata_replace = 0
constant = 0.0001

# Model
# TO BE DEFINED BY USER: path to pretrained backbone weights
pretrained_weights_path = "/dccstor/geofm-finetuning/pretrain_ckpts/mae_weights/2023-04-29_21-50-47/epoch-725-loss-0.0365.pt"
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1

# TRAINING
# iterations = 5000
epochs=250
eval_epoch_interval = 5

# TO BE DEFINED BY USER: Save directory
experiment = "sen1floods11_exp"
work_dir = os.path.join("/dccstor/cimf/paolo/test/", experiment)
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
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
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
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
        split=splits["val"],
    ),
    test=dict(
        type=dataset_type,
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
optimizer = dict(type="Adam", lr=6e-5, weight_decay=0.05)
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

# log_config = dict(
#     interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
    
# )

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', by_epoch=True),
    ])

# This checkpoint config is later overwritten to allow for better logging in mmseg/apis/train.py l. 163
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True, interval=10, out_dir=save_path  # Whether count by epoch or not.
)

evaluation = dict(
    interval=eval_epoch_interval, metric="mIoU", pre_eval=True, save_best="mIoU", by_epoch=True
)


# runner = dict(type="IterBasedRunner", max_iters=iterations)
runner = dict(type="EpochBasedRunner", max_epochs=epochs)

workflow = [("train", 1),("val", 1)]


norm_cfg = dict(type="BN", requires_grad=True)

ce_weights = [0.3, 0.7, 0]

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
        embed_dim=embed_dim,
        output_embed_dim=embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=3,
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
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=ce_weights,
        ),
    ),
    auxiliary_head=dict(
        num_classes=3,
        in_channels=embed_dim,
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
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", stride=(int(tile_size/2), int(tile_size/2)), crop_size=(tile_size, tile_size)),
)
