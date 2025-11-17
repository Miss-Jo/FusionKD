import os
from mmdet.apis import init_detector
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 检查 CUDA
assert torch.cuda.is_available(), "CUDA 不可用！"

# 使用绝对路径
# config = os.path.abspath('configs/crosskd/jojo_glip_res.py')
# checkpoint = os.path.abspath('work_dirs/jojo_glip_res/epoch_1.pth')
config = os.path.abspath('work_dirs/grounding_dino_swin-t_finetune_16xb2_1x_coco/grounding_dino_swin-t_finetune_16xb2_1x_coco.py')
checkpoint = os.path.abspath('work_dirs/grounding_dino_swin-t_finetune_16xb2_1x_coco/epoch_12.pth')

# 加载模型（直接指定设备）
model = init_detector(config, checkpoint, device='cuda:0')

if model is not None:
    model.eval()
    print(f"Params: {count_parameters(model) / 1e6} M")
else:
    print("错误：模型加载失败，请检查配置和权重路径！")
