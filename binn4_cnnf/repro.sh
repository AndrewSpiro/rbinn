#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate cnnf

DEBUG=true
RUN_TRAIN=false
MODEL_DIR='models'

if [ "$DEBUG" = true ]; then
    SAVE_MODEL='CNNF_debug'
    RESULTS_DIR='results_debug'
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100)
    EPOCHS=1
    ATTACK_MODEL='CNN_cifar.pt'
    TARGET_MODEL='CNNF_2_cifar.pt'
else
    SAVE_MODEL='CNNF'
    RESULTS_DIR='results'
    EPOCHS=500
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
fi

if [ "$RUN_TRAIN" = true ]; then
    for T_SEED in "${TRAIN_SEEDS[@]}"
    do
        echo "Running training with seed $T_SEED"

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
                        --seed $T_SEED \
                        --grad-clip \
                        --save-model $SAVE_MODEL \
                        --model-dir $MODEL_DIR \
                        --bool-debug $DEBUG
    done
fi

for A_SEED in "${ATTACK_SEEDS[@]}"
do
    echo "Running attack with seed $A_SEED"

    python test.py --dataset 'cifar10' \
                    --test 'average' \
                    --csv-dir $RESULTS_DIR \
                    --model-dir $MODEL_DIR \
                    --bool-debug $DEBUG \
                    --seed $A_SEED \
                    --attack-model $ATTACK_MODEL \
                    --target-model $TARGET_MODEL
done