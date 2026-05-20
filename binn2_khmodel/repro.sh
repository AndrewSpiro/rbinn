#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate khmodel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname $SCRIPT_DIR)"
echo "Root dir is $ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
DATA_DIR="${ROOT_DIR}/data"
echo "[$SHELL] ## Home data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"

DEBUG=true
TRAIN_LAYER=false
TRAIN_MODEL=true
RUN_ATTACKS=false
TRAIN_MODELS=(khmodel)

if [ "$DEBUG" = true ]; then
    RESULT_DIR="${SCRIPT_DIR}/data/repro/debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=5
    echo "[$SHELL] ## --- RUNNING IN DEBUG MODE ---"
else
    RESULT_DIR="${SCRIPT_DIR}/data/repro"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=1000
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    MODEL_DIR="${RESULT_DIR}/t_seed_${T_SEED}/models"
    FIGURE_DIR="${RESULT_DIR}/t_seed_${T_SEED}/figures"
    EXP_DIR="${RESULT_DIR}/t_seed_${T_SEED}/experiments"
    echo "[$SHELL] ## model dir: ${MODEL_DIR}, figure dir: ${FIGURE_DIR}, exp dir: ${EXP_DIR}"
    mkdir -p $MODEL_DIR $FIGURE_DIR $EXP_DIR
    
    if [ "$TRAIN_LAYER" = true ]; then
        echo "[$SHELL] ## Starting layer training for train seed $T_SEED"
        python src/create_repro.py \
        --epochs $EPOCHS \
        --debug $DEBUG \
        --train_seed $T_SEED \
        --data_path "$DATA_DIR" \
        --model_path $MODEL_DIR \
        --figure_path $FIGURE_DIR \
        --exp_path $EXP_DIR
        
        # convert eps image to png
        gs -dSAFER -dEPSCrop -r600 -sDEVICE=pngalpha -o "${FIGURE_DIR}/FigureA1-FKHL3Spectra.png" "${FIGURE_DIR}/FigureA1-FKHL3Spectra.eps"
    else
        echo "[$SHELL] ## Skipping layer training"
    fi
    if [ "$TRAIN_MODEL" = true ]; then
        echo "[$SHELL] ## Starting model training for train seed $T_SEED"
        python src/train.py \
        --epochs $EPOCHS \
        --train_seed $T_SEED \
        --data_path "$DATA_DIR" \
        --model_path $MODEL_DIR \
        --figure_path $FIGURE_DIR \
        --exp_path $EXP_DIR \
        --num_workers 1 \
        --train_models $TRAIN_MODELS
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
            --figure_path $FIGURE_DIR \
            --exp_path $EXP_DIR \
        done
    fi
done
echo "[$SHELL] ## all experiments completed successfully"