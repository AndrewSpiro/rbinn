#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate pixelreg

DEBUG=true
RUN_TRAIN=false

if [ "$DEBUG" = false]; then
    EPOCHS=1
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    OUTPUT_BASE=output_debug
    RESORE_EPOCH=1
    RESTORE_PATH=restore_debug
    echo "Running in debug mode..."
else
    EPOCHS=70
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    OUTPUT_BASE=output
    RESORE_EPOCH=5
    RESTORE_PATH=restore
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    if [ "$BOOL_TRAIN" = true]; then

        echo "Running training with seed $T_SEED"

        python train.py \
        --dataset cifar10 \
        --in_path ./data \
        --ngpus 1 \
        --epochs $EPOCHS \
        --batch_size 64 \
        -o "${OUTPUT_BASE}_seed_${T_SEED}" \
        -restore_epoch $RESTORE_EPOCH \
        -restore_path $RESTORE_PATH
    fi
done