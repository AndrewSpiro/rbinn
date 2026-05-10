#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate cnnf
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"

TRAIN_METHOD=adv # should be 'adv' or 'clean'
echo "WARNING: so far only 'adv' supported"
DEBUG=false
RUN_TRAIN=true
MODEL_DIR="${SCRIPT_DIR}/models"
BASELINES_PATH="${SCRIPT_DIR}/orig_results.json"
BATCH_SIZE=64

if [ "$DEBUG" = true ]; then
    SAVE_MODEL_BASE="${TRAIN_METHOD}_CNNF_debug"
    RESULTS_DIR_BASE="${SCRIPT_DIR}/${TRAIN_METHOD}_results_debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    EPOCHS=2
    echo "Running in debug mode..."
else
    SAVE_MODEL_BASE="${TRAIN_METHOD}_CNNF"
    RESULTS_DIR_BASE="${SCRIPT_DIR}/${TRAIN_METHOD}_results"
    EPOCHS=500
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
fi


DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
echo "Data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"


for T_SEED in "${TRAIN_SEEDS[@]}"
do
    SAVE_MODEL="${SAVE_MODEL_BASE}_seed_${T_SEED}"

    if [ "$RUN_TRAIN" = true ]; then

        echo "Running training with seed $T_SEED"
        if [ "$TRAIN_METHOD" = adv ]; then
            CLEAN='no'
        else
            echo "WARNING: currently only adversarial training is supported. Supclean is hybrid(?)"
            exit 1
        fi
        
        python "${SCRIPT_DIR}/train.py" \
                        --dataset 'cifar10' \
                        --data-dir "${DATA_DIR}" \
                        --max-cycles 2 \
                        --ind 5 \
                        --mse-parameter 0.1 \
                        --res-parameter 0.1 \
                        --clean $CLEAN \
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
                        --bool-debug $DEBUG \
                        --ckpt_path "${MODEL_DIR}/adv_CNNF_seed_0-epoch149.pt"
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

        python "${SCRIPT_DIR}/test.py" --dataset 'cifar10' \
                        --test 'average' \
                        --data-dir "${DATA_DIR}" \
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
python "${SCRIPT_DIR}/aggregate_results.py" --results_dir $RESULTS_DIR_BASE \
                            --model_name_base $SAVE_MODEL_BASE \
                            --bool_debug $DEBUG \
                            --train_seeds "${TRAIN_SEEDS[@]}" \
                            --attack_seeds "${ATTACK_SEEDS[@]}" \
                            --out $AGG_RESULTS_DIR \
                            --baselines_path $BASELINES_PATH