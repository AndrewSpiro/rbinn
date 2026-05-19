#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate khmodel_clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Root dir is $ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
DATA_DIR="${ROOT_DIR}/data"
echo "[$SHELL] ## Home data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"
MODEL_DIR="${SCRIPT_DIR}/data/repro/models"
echo "[$SHELL] ## Home model dir is here: ${MODEL_DIR}"
mkdir -p "$MODEL_DIR"


DEBUG=false
RUN_TRAINING=true
RUN_ATTACKS=false

if [ "$DEBUG" = true ]; then
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=5
    echo "--- RUNNING IN DEBUG MODE ---"
else
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=1000
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    if [ "$RUN_TRAINING" = true ]; then
        echo "Starting training for train seed $T_SEED"
        python src/create_repro.py \
        --epochs $EPOCHS \
        --debug $DEBUG \
        --train_seed $T_SEED \
        --model_path "$MODEL_DIR" \
        --data_path "$DATA_DIR"
    fi
    if [ "$RUN_ATTACKS" = true ]; then
        echo "Starting attacks for attack seeds $ATTACK_SEEDS"
        for A_SEED in "${ATTACK_SEEDS[@]}"
        do
            echo "Running attack with seed $A_SEED on model $T_SEED"
            python src/run_attacks.py --debug $DEBUG --attack_seed $A_SEED
        done
    fi
done
echo "all experiments completed successfully"