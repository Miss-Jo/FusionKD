_base_ = [
    '../_base_/models/paa_lad_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # student
    # pretrained='torchvision://resnet50',
    # backbone=dict(depth=50),#V2.0直接传递参数
        backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',  # 预训练路径
            # prefix='backbone.'  # 可选：权重前缀对齐
        )
        ),
    # teacher
    # teacher_pretrained="http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth",
    teacher_pretrained="paa_r101_fpn_1x_coco_20200821-0a1825a4.pth",
    teacher_backbone=dict(depth=101),

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),#added by jojo 20250415
)
#optimizer
# optimizer = dict(lr=0.01)
# data = dict(samples_per_gpu=8, workers_per_gpu=4)

#modified by jojo 20250415
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)#12
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=12))
train_dataloader = dict(batch_size=2, num_workers=4)
auto_scale_lr = dict(enable=True, base_batch_size=16)