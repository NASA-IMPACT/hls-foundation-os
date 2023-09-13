#CONFIGURATION PARAMETERS
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
num_frames = 1
img_size = 224
bands = [1, 2, 3, 4, 5, 6]
num_workers = 2
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
checkpoint = ''
gpu_ids = range(0, 1)
dataset_type = 'GeospatialDataset'
find_unused_parameters = True
auto_resume = False
norm_cfg = dict(type='BN', requires_grad=True)



#HYPERPARAMETERS
num_epochs = 100
batch_size = 6
learnr = 6e-4
resume_from = None



#Define the working directory and root directory here
'''
data_root should have two folders conatining labels and inputs respectively

             |- Train -|- Label
data_root ---|         |- Input
             |
             |- Test -|- label
                      |- Input
'''
data_root = '/dccstor/geofm-finetuning/lulc/roi_conus/finetune_data'
cv_num = 1
fraction_data = 1 ## fraction of training data (1,,  0.5, .25. ,  .125)
total_images = 396
mode__ = 'random'
image_dict = {1: 396, 0.5: 198, 0.25: 96, 0.125: 48}
num_images = image_dict[fraction_data]
trainfilename = 'train.txt' if fraction_data ==1 else 'train_{}_{}.txt'.format(int(num_images), cv_num) 
experiment = 'devyani_think_demo/100m_weights_cap_{}epochs_BS_{}_train_{}_mode_{}_lr_{}_CV_{}'.format(num_epochs, batch_size, int(num_images), mode__ , learnr, cv_num) 
work_dir = '/dccstor/geofm-finetuning/lulc/' + experiment 
save_path = work_dir


#OPTIMIZER CONFIG
optimizer = dict(
    type='AdamW',
    lr= learnr,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

#LOG CONFIG
checkpoint_config = dict(
    by_epoch=False,
    interval=  (5*(int(num_images)))//batch_size , # 396/6 *3*5, checkpoint after 5th epoch,
    out_dir= save_path ) 
evaluation = dict(
    interval= (5*(int(num_images)))//batch_size,
     metric='mIoU', pre_eval=True, save_best='mIoU')
runner = dict(type='IterBasedRunner', max_iters= num_epochs* (int(num_images))/batch_size)  ##6600*3) ## 9900 num_epochs* total_images(396_full)/batch_size  =  max_iters  


# INPUT AND LABEL CONFIG
'''
We have used class weights inversely proportional to the data distribution
Class label:Class name
0: No data
1: Water
2: Trees
4: Flooded vegetation
5: Crops
7: Built area
8: Bare ground
9: Snow
10: Clouds
11: Rangeland 
''' 
CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
# mean and std of 6 bands of the inputs
img_norm_cfg = dict(
    type='TorchNormalize',
    means=[472.47, 725.58, 774.52, 2206.44, 1729.96, 1165.04],
    stds=[307.84, 302.57, 333.71, 354.87, 258.31, 222.18])
#size of image
tile_size = 224
orig_nsize = 224
crop_size = (224, 224)
#Weight computed is inversly proportional to the distribution with a cap of 0.2
class_weight = [0,0.0502,0.0152,0,0.2,0.10450532,0,0.0613,0.144,0.2,0.2,0.0242]

#TRAIN PIPELINE
train_pipeline = [
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(
                type='LoadGeospatialAnnotations',
                reduce_zero_label=False,
                nodata=0,
                nodata_replace=0),
            dict(type='BandsExtract', bands=[1, 2, 3, 4, 5, 6]),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
            img_norm_cfg,
            dict(type='Reshape', keys=['img'], new_shape=(6, 1, 224, 224)),
            dict(
                type='Reshape',
                keys=['gt_semantic_seg'],
                new_shape=(1, 224, 224)),
            dict(
                type='CastTensor',
                keys=['gt_semantic_seg'],
                new_type='torch.LongTensor'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]

#VALIDATION PIPELINE
val_pipeline = [
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='BandsExtract', bands=[1, 2, 3, 4, 5, 6]),
            dict(type='ToTensor', keys=['img']),
            img_norm_cfg,
            dict(type='AddTimeDimension'),
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
        ]

#TEST PIPELINE
test_pipeline = [
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='BandsExtract', bands=[1, 2, 3, 4, 5, 6]),
            dict(type='ToTensor', keys=['img']),
            img_norm_cfg,
            dict(type='AddTimeDimension'),
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
        ]

data = dict(
    samples_per_gpu= batch_size,
    workers_per_gpu=1,
    train=dict(
        type='GeospatialDataset',
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir='train_small/HLS.L30',
        ann_dir='train_small/LUC.S30',
        pipeline= train_pipeline,
        img_suffix='HLS.L30.tif',
        seg_map_suffix='LUC.S30.tif',
        ignore_index=0,
        split=data_root+'/splits/' + trainfilename
    ),
    val=dict(
        type='GeospatialDataset',
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir='train_small/HLS.L30',
        ann_dir='train_small/LUC.S30',
        pipeline=val_pipeline,
        img_suffix='HLS.L30.tif',
        seg_map_suffix='LUC.S30.tif',
        ignore_index=0,
        split=data_root+'/splits/validation.txt'
    ),
    test=dict(
        type='GeospatialDataset',
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir='test_small/HLS.L30',
        ann_dir='test_small/LUC.S30',
        pipeline=test_pipeline,
        img_suffix='HLS.L30.tif',
        seg_map_suffix='LUC.S30.tif',
        ignore_index=0,
        split=data_root+'/splits/test.txt',
        gt_seg_map_loader_cfg=dict(nodata=0, nodata_replace=0))
        )

#MODEL ARCHITECTURE
model = dict(
    type='TemporalEncoderDecoder',
    pretrained= '/dccstor/geofm-finetuning/pretrain_ckpts/mae_weights/2023-04-29_21-50-47/epoch-725-loss-0.0365.pt',
    backbone=dict(
        type='PretrainVisionTransformer',
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    decode_head=dict(
        num_classes=12,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight = class_weight)),
    auxiliary_head=dict(
        num_classes=12,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1, class_weight = class_weight)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(128, 128), crop_size=(224, 224)))

