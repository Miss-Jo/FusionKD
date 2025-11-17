
_base_ = [
    '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_2x.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

lang_model_name = 'bert-base-uncased'
teacher_ckpt="checkpoint/glip_tiny_model_o365_goldg_cc_sbu.pth"

data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32)
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='FusionKDATSS',

    data_preprocessor=data_preprocessor,
    teacher_config='configs/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco.py',

    backbone=dict(
        type='ResNet',
        depth=50,#50
        # depth=101,#50
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        # num_classes=10,#80
        num_classes=6, ## 必须等于数据集类别数量
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
     kd_cfg=dict(
        loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),#val normal,result:42.0
        loss_reg_kd=dict(type='GIoULoss', loss_weight=1.0),#LOSS_WEIGHT 1.0
        loss_center_kd=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),#LOSS_WEIGHT 1.0
        loss_feat_kd=dict(type='PKDLoss', loss_weight=0.5), #3.0 ！！！
        reused_teacher_head_idx=3
    ),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
       nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
    )




train_dataloader = dict(batch_size=2, num_workers=4)#8  2(yes!) 4(OOD)
auto_scale_lr = dict(enable=True, base_batch_size=4)


# training schedule
max_epochs =12#120 33
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# 自定义Hook配置
custom_hooks = [
    dict(
        type='KDWeightUpdateHook',
        update_interval=1,
        strategy='adaptive'  # 可选: 'adaptive', 'cosine''performance_aware',
    ),
    dict(type='GradientCoordinatorHook'),
    # 其他原有hooks...
]

# 学习率调度优化
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         end_factor=1.0,
#         begin=0,
#         end=5,
#         by_epoch=True
#     ),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=25,
#         eta_min=1e-6,
#         begin=5,
#         end=30,
#         by_epoch=True
#     )
# ]
# # 回退到接近原始配置，只做最小修改
# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=20,  # 延长衰减周期
#         eta_min=5e-06,  # 仅提高最低学习率
#         by_epoch=True,
#         begin=12,  # 推迟开始衰减
#         end=30
#     )
# ]
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=1e-07,
        by_epoch=True,
        begin=10,
        end=30)
]


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None)#ours