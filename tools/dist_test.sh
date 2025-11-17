#!/usr/bin/env bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

CUDA_VISIBLE_DEVICES=0,1,2,3
#CONFIG="configs/crosskd/crosskd_r50_gflv1_r101-2x-ms_fpn_1x_coco.py"
#CONFIG="configs/retinanet/retinanet_swin-t-p4-w7_fpn_1x_coco.py"
#CONFIG="configs/crosskd/crosskd_r50_retinanet_r101_fpn_2x_coco.py"
#CONFIG="configs/crosskd/crosskd_r50_fcos_r101-2x-ms_caffe_fpn_gn-head_2x_ms_coco.py"
#CONFIG="configs/crosskd/crosskd_r50_atss_r101_fpn_1x_coco.py"
#CONFIG="configs/crosskd/crosskd_r50_retinanet_swint_fpn_1x_coco.py"

#CONFIG="configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py"
#CHECKPOINT="work_dirs/yolov3_mobilenetv2_8xb24-ms-416-300e_coco/epoch_13.pth"

CONFIG="configs/crosskd/jojo_glip_res.py"
CHECKPOINT="work_dirs/jojo_glip_res/fusionkd_50_77.5_consine_no_feature/epoch_30.pth"

GPUS=4

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#usage: test.py [-h] [--work-dir WORK_DIR] [--out OUT] [--show] [--show-dir SHOW_DIR] [--wait-time WAIT_TIME] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--launcher {none,pytorch,slurm,mpi}] [--tta]
#               [--local_rank LOCAL_RANK]
#               config checkpoint

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
#python tools/analysis_tools/benchmark.py  configs/crosskd/jojo_glip_res.py --checkpoint work_dirs/jojo_glip_res/fusionkd_50_77.5_consine_no_feature/epoch_30.pth