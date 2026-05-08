#!/bin/bash

# Script to evaluate and compare two trained models
# Usage: ./compare_model_performance.sh

set -e

# Define experiments
EXP1_DIR="work_dirs/lanesegnet_8e_carla_24e_olv2_subset_A_ls"
EXP1_NAME="2-stage (subset_A_ls)"
EXP1_CONFIG="$EXP1_DIR/lanesegnet_r50_8x1_24e_olv2_subset_A.py"
EXP1_CKPT="$EXP1_DIR/epoch_24.pth"

EXP2_DIR="work_dirs/lanesegnet_8e_carla_laneseg_text_guided_1031"
EXP2_NAME="Text-guided (1031)"
EXP2_CONFIG="$EXP2_DIR/lanesegnet_r50_1x2_24e_olv2_subset_A_text_guided_stage2.py"
EXP2_CKPT="$EXP2_DIR/epoch_24.pth"

echo "=========================================="
echo "Evaluating Models"
echo "=========================================="
echo ""

# Evaluate Experiment 1
echo "Evaluating $EXP1_NAME..."
python tools/test.py \
    $EXP1_CONFIG \
    $EXP1_CKPT \
    --eval-options jsonfile_prefix=$EXP1_DIR/results \
    | tee $EXP1_DIR/eval_results.log

echo ""
echo "Evaluating $EXP2_NAME..."
python tools/test.py \
    $EXP2_CONFIG \
    $EXP2_CKPT \
    --eval-options jsonfile_prefix=$EXP2_DIR/results \
    | tee $EXP2_DIR/eval_results.log

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - $EXP1_DIR/eval_results.log"
echo "  - $EXP2_DIR/eval_results.log"
echo ""
echo "Run the following to compare results:"
echo "  python tools/compare_experiments.py"
