#!/bin/bash

#SBATCH --job-name=b2_test
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="interactive"
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=2G

set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate khmodel

# Home directories
SCRIPT_DIR=$PWD
PARENT_DIR="$(dirname $SCRIPT_DIR)"
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
HOME_DATA_DIR="$PARENT_DIR/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
mkdir -p "$HOME_DATA_DIR"
HOME_MODEL_DIR="${SCRIPT_DIR}/data/repro/models"
echo "[$SHELL] ## Home model dir is here: ${HOME_MODEL_DIR}"
mkdir -p "$HOME_MODEL_DIR"
FIGURE_DIR="${SCRIPT_DIR}/data/repro/figures"

# Scratch directories for fast loading
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
SCRATCH_DATA_DIR="${SCRATCH_PATH}/data"
SCRATCH_MODEL_DIR="${SCRATCH_PATH}/models"
mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_MODEL_DIR"

# Copy data from home to scratch
cp -r "$HOME_DATA_DIR/" "$SCRATCH_PATH/"
echo "[$SHELL] ## Data copied to scratch"
echo "[$SHELL] ## Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

#### Experiment pipeline ####
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
    MODEL_DIR="${SCRATCH_MODEL_DIR}/model_seed_${T_SEED}"
    if [ "$RUN_TRAINING" = true ]; then
        echo "[$SHELL] ## Starting training for train seed $T_SEED"
        python src/create_repro.py \
            --epochs $EPOCHS \
            --debug $DEBUG \
            --train_seed $T_SEED \
            --model_path $MODEL_DIR \
            --data_path "${SCRATCH_DATA_DIR}" \
            --figure_path $FIGURE_DIR

        # copy models from scratch to home
        cp -r $MODEL_DIR $HOME_MODEL_DIR

        # convert eps image to png
        gs -dSAFER -dEPSCrop -r600 -sDEVICE=pngalpha -o "${FIGURE_DIR}/FigureA1-FKHL3Spectra.png" "${FIGURE_DIR}/FigureA1-FKHL3Spectra.eps"
    fi

    # copy models from home to scratch
    cp -r "${HOME_MODEL_DIR}/model_seed_${T_SEED}" $SCRATCH_MODEL_DIR
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
            --khlayer_name fkhl3_cifar10_pruned.pty
        done
    fi
done
echo "[$SHELL] ## all experiments completed successfully"