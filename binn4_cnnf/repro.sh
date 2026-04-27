#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate pixelreg
DEBUG=true

if [ "$DEBUG" = true ]; then
SAVE_MODEL='CNNF_debug'
EPOCHS=1
else
EPOCHS=500
SAVE_MODEL='CNNF'
fi

python train.py --data 'cifar10' \
                --max-cycles 2 \
                --ind 5 \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --clean 'supclean' \
                --clean-parameter 0.05 \
                --lr 0.05 \
                --batch-size 64 \
                --eps 0.063 \
                --eps-iter 0.02 \
                --schedule 'poly' \
                --epochs $EPOCHS \
                --seed 0 \
                --grad-clip \
                --save-model $SAVE_MODEL \
                --model-dir 'models'


