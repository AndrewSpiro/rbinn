#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate cnnf

DEBUG=true
RUN_TRAIN=false
MODEL_DIR='models'
BASELINES_PATH=orig_results.json
BATCH_SIZE=64

if [ "$DEBUG" = true ]; then
    SAVE_MODEL_BASE='CNNF_debug'
    RESULTS_DIR_BASE='results_debug'
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    EPOCHS=1
    echo "Running in debug mode..."
else
    SAVE_MODEL_BASE='CNNF'
    RESULTS_DIR_BASE='results'
    EPOCHS=500
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
fi


for T_SEED in "${TRAIN_SEEDS[@]}"
do
    SAVE_MODEL="${SAVE_MODEL_BASE}_seed_${T_SEED}"

    if [ "$RUN_TRAIN" = true ]; then

        echo "Running training with seed $T_SEED"
        
        python train.py --data 'cifar10' \
                        --max-cycles 2 \
                        --ind 5 \
                        --mse-parameter 0.1 \
                        --res-parameter 0.1 \
                        --clean 'supclean' \
                        --clean-parameter 0.05 \
                        --lr 0.05 \
                        --batch-size $BATCH_SIZE \
                        --eps 0.063 \
                        --eps-iter 0.02 \
                        --schedule 'poly' \
                        --epochs $EPOCHS \
                        --seed $T_SEED \
                        --grad-clip \
                        --save-model $SAVE_MODEL \
                        --model-dir $MODEL_DIR \
                        --bool-debug $DEBUG
    fi        

    for A_SEED in "${ATTACK_SEEDS[@]}"
    do
        if [ "$DEBUG" = true ]; then
            # ATTACK_MODEL='CNN_cifar'
            TARGET_MODEL=$SAVE_MODEL
        else
            TARGET_MODEL=$SAVE_MODEL
        fi

        echo "Running attack with seed $A_SEED"

        RESULTS_DIR="${RESULTS_DIR_BASE}/${TARGET_MODEL}/attack_seed_${A_SEED}"
        mkdir -p "$RESULTS_DIR"

        python test.py --dataset 'cifar10' \
                        --test 'average' \
                        --results-path "${RESULTS_DIR}/results.json" \
                        --model-dir $MODEL_DIR \
                        --bool-debug $DEBUG \
                        --seed $A_SEED \
                        --target-model "${TARGET_MODEL}.pt"
    done
done

TRAIN_SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
ATTACK_SEED_STRING=$(echo "${ATTACK_SEEDS[*]}" | tr ' ' '_')
AGG_RESULTS_DIR="${RESULTS_DIR_BASE}/train_seeds_${TRAIN_SEED_STRING}/attack_seeds_${ATTACK_SEED_STRING}"
mkdir -p "$AGG_RESULTS_DIR"
python aggregate_results.py --results_dir $RESULTS_DIR_BASE \
                            --model_name_base $SAVE_MODEL_BASE \
                            --bool_debug $DEBUG \
                            --train_seeds "${TRAIN_SEEDS[@]}" \
                            --attack_seeds "${ATTACK_SEEDS[@]}" \
                            --out $AGG_RESULTS_DIR \
                            --baselines_path $BASELINES_PATH