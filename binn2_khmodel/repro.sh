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
HOME_MODEL_DIR="${SCRIPT_DIR}/data/repro/models"
echo "[$SHELL] ## Home model dir is here: ${HOME_MODEL_DIR}"
mkdir -p "$HOME_MODEL_DIR"
FIGURE_DIR="${SCRIPT_DIR}/data/repro/figures"


DEBUG=false
RUN_TRAINING=false
RUN_ATTACKS=true

if [ "$DEBUG" = true ]; then
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=5
    echo "[$SHELL] ## --- RUNNING IN DEBUG MODE ---"
else
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=1000
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    MODEL_DIR="${HOME_MODEL_DIR}/model_seed_${T_SEED}"
    if [ "$RUN_TRAINING" = true ]; then
        echo "[$SHELL] ## Starting training for train seed $T_SEED"
        python src/create_repro.py \
        --epochs $EPOCHS \
        --debug $DEBUG \
        --train_seed $T_SEED \
        --model_path $MODEL_DIR \
        --data_path "$DATA_DIR" \
        --figure_path $FIGURE_DIR
        
        # convert eps image to png
        gs -dSAFER -dEPSCrop -r600 -sDEVICE=pngalpha -o "${FIGURE_DIR}/FigureA1-FKHL3Spectra.png" "${FIGURE_DIR}/FigureA1-FKHL3Spectra.eps"
    fi
    if [ "$RUN_ATTACKS" = true ]; then
        echo "[$SHELL] ## Starting attacks for attack seeds $ATTACK_SEEDS"
        for A_SEED in "${ATTACK_SEEDS[@]}"
        do
            mkdir -p "${FIGURE_DIR}/attack_seed_${A_SEED}"
            echo "[$SHELL] ## Running attack with seed $A_SEED on model $T_SEED"
            python src/run_attacks.py \
            --debug $DEBUG \
            --attack_seed $A_SEED \
            --data_path $DATA_DIR \
            --model_path $MODEL_DIR \
            --model_name fkhl3_cifar10_pruned.pty
        done
    fi
done
echo "[$SHELL] ## all experiments completed successfully"