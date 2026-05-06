#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate vonenet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$SCRIPT_DIR")"
PACKAGE="$(basename "$SCRIPT_DIR")"
cd "$PARENT"

DEBUG=true
RUN_TRAIN=true
RUN_EVAL=true
MODEL_ARCH=resnet50

if [ "$DEBUG" = true ]; then
    EPOCHS=0
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    RESTORE_EPOCH=1
    ROOT="${SCRIPT_DIR}/debug"
    echo "Running in debug mode..."
else
    EPOCHS=70
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    RESTORE_EPOCH=5
    ROOT="${SCRIPT_DIR}"
fi

DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
echo "Data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"

for T_SEED in "${TRAIN_SEEDS[@]}"
do

    OUTPUT_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/output"
    mkdir -p $OUTPUT_PATH
    
    RESTORE_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/restore"
    mkdir -p $RESTORE_PATH
    
    RESULTS_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/results.json"

    if [ "$RUN_TRAIN" = true ]; then

        echo "Running training with seed $T_SEED"

        python -m "$PACKAGE.train" train \
        --dataset cifar10 \
        --in_path $DATA_DIR \
        --ngpus 1 \
        --epochs $EPOCHS \
        --batch_size 64 \
        -o $OUTPUT_PATH \
        -restore_path $RESTORE_PATH \
        --model_arch $MODEL_ARCH

        echo 'Finished training.'
    fi

    if [ "$RUN_EVAL" = true ]; then
        python -m "$PACKAGE.run" \
        --in_path $DATA_DIR \
        --ngpus 1 \
        --vonenet_dir $OUTPUT_PATH \
        --epoch $EPOCHS \
        --results_path $RESULTS_PATH
    fi
done

TRAIN_SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
AGG_RESULTS_DIR="${ROOT}/aggregated_results/${MODEL_ARCH}_train_seeds_${TRAIN_SEED_STRING}"
python -m "$PACKAGE.aggregate_results" --root $ROOT \
                            --train_seeds "${TRAIN_SEEDS[@]}" \
                            --model_arch $MODEL_ARCH \
                            --out $AGG_RESULTS_DIR
