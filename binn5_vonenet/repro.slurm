#!/bin/bash

#SBATCH --job-name=b5_adv
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-l4-24g"
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2G

set -e

source $(conda info --base)/etc/profile.d/conda.sh

conda activate vonenet

# Home directories
SCRIPT_DIR=$PWD
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
HOME_DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
mkdir -p "$HOME_DATA_DIR"
# Copy data from home to scratch
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
cp -r "$HOME_DATA_DIR" "$SCRATCH_PATH/"
echo "[$SHELL] ## Data copied to scratch"
SCRATCH_DATA_DIR="${SCRATCH_PATH}/$(basename $HOME_DATA_DIR)"
echo "[$SHELL] ## Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

DEBUG=true
RUN_TRAIN=true
RUN_EVAL=true
MODEL_ARCH=resnet50
TRAIN_METHOD='clean'
TRAIN_EPSILON=0.063

if [ "$DEBUG" = true ]; then
    EPOCHS=0
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    RESTORE_EPOCH=1
    HOME_ROOT="${SCRIPT_DIR}/debug"
    ROOT="${SCRATCH_PATH}/debug"
    echo "Running in debug mode..."
else
    EPOCHS=70
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100 101 102)
    RESTORE_EPOCH=5
    HOME_ROOT="${SCRIPT_DIR}"
    ROOT="${SCRATCH_PATH}"
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do

    OUTPUT_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/output"
    mkdir -p $OUTPUT_PATH
    
    RESTORE_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/restore"
    mkdir -p $RESTORE_PATH
    
    RESULTS_PATH="${ROOT}/${MODEL_ARCH}_vonenet_seed_${T_SEED}/results.json"

    if [ "$RUN_TRAIN" = true ]; then

        echo "Running training with seed $T_SEED"

        python "${SCRIPT_DIR}/train.py" train \
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

    cp -r $OUTPUT_PATH $RESTORE_PATH $RESULTS_PATH $HOME_ROOT

    if [ "$RUN_EVAL" = true ]; then
        python "${SCRIPT_DIR}/run.py" \
        --in_path $DATA_DIR \
        --ngpus 1 \
        --vonenet_dir $OUTPUT_PATH \
        --epoch $EPOCHS \
        --results_path $RESULTS_PATH
    fi

    cp -r $OUTPUT_PATH $RESTORE_PATH $RESULTS_PATH $HOME_ROOT

done

TRAIN_SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
AGG_RESULTS_DIR="${ROOT}/aggregated_results/${MODEL_ARCH}_train_seeds_${TRAIN_SEED_STRING}"
python "${SCRIPT_DIR}/aggregate_results.py" --root $ROOT \
                            --train_seeds "${TRAIN_SEEDS[@]}" \
                            --model_arch $MODEL_ARCH \
                            --out $AGG_RESULTS_DIR

cp -r $AGG_RESULTS_DIR $HOME_ROOT