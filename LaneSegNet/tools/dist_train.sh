#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`
export PYTHONPATH="/home/LaneSegNet/":$PYTHONPATH

WORK_DIR=work_dirs/ablation_text_supervision_stage2
CONFIG=projects/configs/ablation_singlehead_supervision_stage2.py
GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
