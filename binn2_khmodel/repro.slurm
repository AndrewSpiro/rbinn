#!/bin/bash

#SBATCH --job-name=b2_test
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="interactive"
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=2G

set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate khmodel

SCRIPT_DIR=$PWD
ROOT_DIR="$(dirname $SCRIPT_DIR)"
echo "Root dir is $ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
HOME_DATA_DIR="${ROOT_DIR}/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
mkdir -p "$HOME_DATA_DIR"
# Copy data from home to scratch
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
cp -r "$HOME_DATA_DIR" "$SCRATCH_PATH/"
echo "[$SHELL] ## Data copied to scratch"
SCRATCH_DATA_DIR="${SCRATCH_PATH}/$(basename $HOME_DATA_DIR)"
echo "[$SHELL] ## Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

DEBUG=true
TRAIN_LAYER=false
TRAIN_MODEL=true
RUN_ATTACKS=false
TRAIN_MODELS=(khmodel)

if [ "$DEBUG" = true ]; then
    HOME_RESULT_DIR="${SCRIPT_DIR}/data/repro/debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=5
    echo "[$SHELL] ## --- RUNNING IN DEBUG MODE ---"
else
    HOME_RESULT_DIR="${SCRIPT_DIR}/data/repro"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    EPOCHS=1000
fi

echo "[$SHELL] ## Home result dir is here: ${HOME_RESULT_DIR}"
mkdir -p "$HOME_RESULT_DIR"
# Copy data from home to scratch
cp -r "$HOME_RESULT_DIR" "$SCRATCH_PATH/"
echo "[$SHELL] ## results copied to scratch"
SCRATCH_RESULT_DIR="${SCRATCH_PATH}/$(basename $HOME_RESULT_DIR)"
echo "[$SHELL] ## Contents of scratch result dir: $(ls -R $SCRATCH_RESULT_DIR)"

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    MODEL_DIR="${SCRATCH_RESULT_DIR}/t_seed_${T_SEED}/models"
    FIGURE_DIR="${SCRATCH_RESULT_DIR}/t_seed_${T_SEED}/figures"
    EXP_DIR="${SCRATCH_RESULT_DIR}/t_seed_${T_SEED}/experiments"
    echo "[$SHELL] ## model dir: ${MODEL_DIR}, figure dir: ${FIGURE_DIR}, exp dir: ${EXP_DIR}"
    mkdir -p $MODEL_DIR $FIGURE_DIR $EXP_DIR

    if [ "$TRAIN_LAYER" = true ]; then
        echo "[$SHELL] ## Starting layer training for train seed $T_SEED"
        python src/create_repro.py \
            --epochs $EPOCHS \
            --debug $DEBUG \
            --train_seed $T_SEED \
            --data_path "${SCRATCH_DATA_DIR}" \
            --model_path $MODEL_DIR \
            --figure_path $FIGURE_DIR \
            --exp_path $EXP_DIR

        # copy models from scratch to home
        cp -r $MODEL_DIR $FIGURE_DIR $HOME_RESULT_DIR

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
        --data_path "$SCRATCH_DATA_DIR" \
        --model_path $MODEL_DIR \
        --figure_path $FIGURE_DIR \
        --exp_path $EXP_DIR \
        --num_workers 1 \
        --train_models $TRAIN_MODELS

        # copy models from scratch to home
        cp -r $MODEL_DIR $HOME_RESULT_DIR
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
            --data_path $SCRATCH_DATA_DIR \
            --model_path $MODEL_DIR \
            --figure_path "${FIGURE_DIR}/a_seed_${A_SEED}" \
            --exp_path "${EXP_DIR}/a_seed_${A_SEED}"
        done

        # copy models from scratch to home
        cp -r $FIGURE_DIR $EXP_DIR $HOME_RESULT_DIR
    fi
done
echo "[$SHELL] ## all experiments completed successfully"