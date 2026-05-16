#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate khmodel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

DEBUG=true
RUN_TRAINING=true
RUN_ATTACKS=true

if [ "$DEBUG" = true ]; then
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=5
    echo "--- RUNNING IN DEBUG MODE ---"
else
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    EPOCHS=1000
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    if [ "$RUN_TRAINING" = true ]; then
        echo "Starting training for train seed $T_SEED"
        python src/create_repro.py --epochs $EPOCHS --debug $DEBUG
    fi
    if [ "$RUN_ATTACKS" = true ]; then
        echo "Starting attacks for attack seeds $ATTACK_SEEDS"
        for A_SEED in "${ATTACK_SEEDS[@]}"
        do
            echo "Running attack with seed $A_SEED on model $T_SEED"
        done
    fi
done
