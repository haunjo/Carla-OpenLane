#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`
export PYTHONPATH="/home/user/LaneSegNet/":$PYTHONPATH

WORK_DIR=work_dirs/lanesegnet_baseline_mapele_bucket
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py
CHECKPOINT=${WORK_DIR}/latest.pth

GPUS=${1:-1}
PORT=${PORT:-28510}

# Run test with visualization
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/vis_test --eval openlane_v2 --show --show-dir ${WORK_DIR}/vis_results \
    2>&1 | tee ${WORK_DIR}/vis_test.${timestamp}.log
