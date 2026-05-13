#!/bin/bash

#SBATCH --job-name=b4_clean
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="testing"
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate cnnf

# Home directories
SCRIPT_DIR=$PWD
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
HOME_DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
mkdir -p "$HOME_DATA_DIR"
HOME_MODEL_DIR="${SCRIPT_DIR}/models"
echo "[$SHELL] ## Home model dir is here: ${HOME_MODEL_DIR}"
mkdir -p "$HOME_MODEL_DIR"

# Scratch directories for fast loading
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
SCRATCH_DATA_DIR="${SCRATCH_PATH}/data"
SCRATCH_MODEL_DIR="${SCRATCH_PATH}/models"
mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_MODEL_DIR"

# Copy data from home to scratch
cp -r "$HOME_DATA_DIR/" "$SCRATCH_DATA_DIR/"
echo "[$SHELL] ## Data copied to scratch"
echo "Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

#### Experiment pipeline ####
TRAIN_METHOD=clean
DEBUG=true
RUN_TRAIN=true
BASELINES_PATH="${SCRIPT_DIR}/orig_results.json"
BATCH_SIZE=64

if [ "$DEBUG" = true ]; then
    SAVE_MODEL_BASE="${TRAIN_METHOD}_CNNF_debug" # This is not a path- just a name that will be appended to a path
    RESULTS_DIR_BASE="${SCRIPT_DIR}/${TRAIN_METHOD}_results_debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    EPOCHS=1
    echo "[$SHELL] ## Running in debug mode..."
else
    SAVE_MODEL_BASE="${TRAIN_METHOD}_CNNF"
    RESULTS_DIR_BASE="${SCRIPT_DIR}/${TRAIN_METHOD}_results"
    EPOCHS=500
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
fi


for T_SEED in "${TRAIN_SEEDS[@]}"
do
    SAVE_MODEL="${SAVE_MODEL_BASE}_seed_${T_SEED}"

    if [ "$RUN_TRAIN" = true ]; then

        echo "[$SHELL] ## Running $TRAIN_METHOD training with seed $T_SEED"
        if [ "$TRAIN_METHOD" = adv ]; then
            CLEAN=no

        elif [ "$TRAIN_METHOD" = clean ]
        then
            CLEAN=only
        else
            CLEAN=supclean
            echo "[$SHELL] ## WARNING: Training with supclean. This is a clean-adversarial hybrid and probably not the desired method."
        fi

        python "${SCRIPT_DIR}/train.py" \
                        --dataset 'cifar10' \
                        --data-dir "${SCRATCH_DATA_DIR}" \
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
                        --model-dir $SCRATCH_MODEL_DIR \
                        --bool-debug $DEBUG \
                        --num_workers 1
    fi        

    for A_SEED in "${ATTACK_SEEDS[@]}"
    do
        if [ "$DEBUG" = true ]; then
            # ATTACK_MODEL='CNN_cifar'
            TARGET_MODEL=$SAVE_MODEL
        else
            TARGET_MODEL=$SAVE_MODEL
        fi

        echo "[$SHELL] ## Running attack with seed $A_SEED"

        RESULTS_DIR="${RESULTS_DIR_BASE}/${TARGET_MODEL}/attack_seed_${A_SEED}"
        mkdir -p "$RESULTS_DIR"

        python "${SCRIPT_DIR}/test.py" --dataset 'cifar10' \
                        --test 'average' \
                        --data-dir "${SCRATCH_DATA_DIR}" \
                        --results-path "${RESULTS_DIR}/results.json" \
                        --model-dir $SCRATCH_MODEL_DIR \
                        --bool-debug $DEBUG \
                        --seed $A_SEED \
                        --target-model "${TARGET_MODEL}.pt" \
                        --num_workers 1
    done
done

cp -r $SCRATCH_MODEL_DIR $HOME_MODEL_DIR
echo "[$SHELL] ## Model copied to home"

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