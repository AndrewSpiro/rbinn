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

DEBUG=false
TRAIN_LAYER=false
TRAIN_MODEL=false
RUN_ATTACKS=true
AGG_RESULTS=false
TRAIN_MODELS=(khmodel)
TRAIN_ATTACK=fgsm # set to clean for clean training
TRAIN_EPSILON_NUMERATOR=16
TRAIN_EPSILON="${TRAIN_EPSILON_NUMERATOR}/255"
ACC_TYPE=relative # either absolute or relative

if [ "$DEBUG" = true ]; then
    RESULT_DIR="${SCRIPT_DIR}/data/repro/debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(102 103)
    EPOCHS=100
    NUM_EPS=5
    echo "[$SHELL] ## --- RUNNING IN DEBUG MODE ---"
else
    RESULT_DIR="${SCRIPT_DIR}/data/repro"
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(101 102 103)
    EPOCHS=1000
    NUM_EPS=400
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do

    if [ "${TRAIN_ATTACK}" = clean ]; then
        TRAIN_CONFIG_DIR="${TRAIN_ATTACK}"
    else
        TRAIN_CONFIG_DIR="${TRAIN_ATTACK}_${TRAIN_EPSILON_NUMERATOR}"
    fi

    MODEL_DIR="${RESULT_DIR}/${TRAIN_CONFIG_DIR}/t_seed_${T_SEED}/models"
    FIGURE_DIR="${RESULT_DIR}/${TRAIN_CONFIG_DIR}/t_seed_${T_SEED}/figures"
    EXP_DIR="${RESULT_DIR}/${TRAIN_CONFIG_DIR}/t_seed_${T_SEED}/experiments/${ACC_TYPE}"
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
        --train_models "${TRAIN_MODELS[@]}" \
        --train_attack $TRAIN_ATTACK \
        --train_epsilon $TRAIN_EPSILON
    else
        echo "[$SHELL] ## Skipping model training"
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
            --figure_path "${FIGURE_DIR}/a_seed_${A_SEED}" \
            --exp_path "${EXP_DIR}/a_seed_${A_SEED}" \
            --attack_models "${TRAIN_MODELS[@]}" \
            --num_eps $NUM_EPS \
            --acc_type $ACC_TYPE
        done
    else
        echo "[$SHELL] ## Skipping attacks"
    fi

    if [ "$AGG_RESULTS" = true ]; then
        python src/aggregate_results.py \
            --result_path $RESULT_DIR
    else
        echo "[$SHELL] ## Skipping aggregating results"
    fi
done
echo "[$SHELL] ## all experiments completed successfully"