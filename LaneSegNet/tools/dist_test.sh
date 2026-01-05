#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`
export PYTHONPATH="/home/LaneSegNet/":$PYTHONPATH

WORK_DIR=work_dirs/lanesegnet_2stage_text_guided_mapele_bucket
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py

CHECKPOINT=${WORK_DIR}/epoch_24.pth
# CHECKPOINT=./lanesegnet_r50_2x2_24e_olv2_subset_A_mapele_bucket.pth

GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
