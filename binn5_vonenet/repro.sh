#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate vonenet

DEBUG=true
RUN_TRAIN=true
RUN_EVAL=true
MODEL_ARCH=resnet50

if [ "$DEBUG" = true ]; then
    EPOCHS=1
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    OUTPUT_BASE=output_debug
    RESTORE_EPOCH=1
    RESTORE_BASE=restore_debug
    RESULTS_DIR=debug_results
    echo "Running in debug mode..."
else
    EPOCHS=70
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    OUTPUT_BASE=output
    RESTORE_EPOCH=5
    RESTORE_BASE=restore
    RESULTS_DIR=results
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    if [ "$RUN_TRAIN" = true ]; then

        echo "Running training with seed $T_SEED"

        mkdir -p $OUTPUT_BASE
        VONENET_DIR="${OUTPUT_BASE}/${MODEL_ARCH}_vonenet_seed_${T_SEED}"
        
        mkdir -p $RESTORE_BASE
        
        mkdir -p "${RESULTS_DIR}/${MODEL_ARCH}_seed_${T_SEED}"
        RESULTS_PATH="${RESULTS_DIR}/${MODEL_ARCH}_seed_${T_SEED}/results.json"

        python train.py train \
        --dataset cifar10 \
        --in_path ../data \
        --ngpus 1 \
        --epochs $EPOCHS \
        --batch_size 64 \
        -o "${OUTPUT_BASE}/${MODEL_ARCH}_vonenet_seed_${T_SEED}" \
        -restore_path "$RESTORE_BASE/${MODEL_ARCH}_vonenet_seed_${T_SEED}" \
        --model_arch $MODEL_ARCH \
        --vonenet_dir $VONENET_DIR

        echo 'Finished training.'
    fi

    if [ "$RUN_EVAL" = true ]; then
        python run.py \
        --in_path ../data \
        --ngpus 1 \
        --vonenet_dir $VONENET_DIR \
        --results_path $RESULTS_PATH
    fi
done

TRAIN_SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
AGG_RESULTS_DIR="${RESULTS_DIR}/${MODEL_ARCH}_train_seeds_${TRAIN_SEED_STRING}"
python aggregate_results.py --results_dir $RESULTS_DIR \
                            --train_seeds "${TRAIN_SEEDS[@]}" \
                            --model_name_base $MODEL_ARCH \
                            --out $AGG_RESULTS_DIR
