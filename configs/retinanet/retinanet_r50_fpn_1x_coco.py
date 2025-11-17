_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

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
