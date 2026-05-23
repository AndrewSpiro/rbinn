#!/bin/bash

#SBATCH --job-name=b1_val
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="testing"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate pixelreg
SCRIPT_DIR=$PWD

ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"

DEBUG=true
RUN_TRAINING=true
TRAIN_ATTACK=fgsm
TRAIN_EPSILON=8/255
RUN_ATTACKS=true

TASK=CIFAR10
ARCHI=ResNet18
REG_DATA=CIFAR10
REG_ALPHA=10
REG_THRESH=0.8
RGB=true
ATTACK=false
ATTACK_TYPES=(Gaussian)
GAUSSIAN_RANGE="0.0 0.3"

if [ "$DEBUG" = true ]; then
    EPOCHS=0
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(101)
    HOME_SAVE_DIR="${SCRIPT_DIR}/save/debug"
    echo "--- RUNNING IN DEBUG MODE ---"
else
    EPOCHS=40
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    HOME_SAVE_DIR="${SCRIPT_DIR}/save"
fi

echo "[$SHELL] ## Home save dir is here: ${HOME_SAVE_DIR}"
cp -r $HOME_SAVE_DIR "$SCRATCH_PATH/"
echo "[$SHELL] ## Saves copied to scratch"
SAVE_DIR="${SCRATCH_PATH}/$(basename $HOME_SAVE_DIR)"
echo "[$SHELL] ## Contents of save dir: $(ls $SAVE_DIR)"

HOME_DATA_DIR="${ROOT_DIR}/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
cp -r "$HOME_DATA_DIR" "$SCRATCH_PATH/"
echo "[$SHELL] ## Data copied to scratch"
DATA_DIR="${SCRATCH_PATH}/$(basename $HOME_DATA_DIR)"
echo "[$SHELL] ## Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    MODEL_DIR="${SAVE_DIR}/model_seed_$T_SEED"
    mkdir -p "$MODEL_DIR"

    if [ "$RUN_TRAINING" = true ]; then
        #Train
        echo "Starting training for train seed $T_SEED"

        python "${SCRIPT_DIR}/train.py" \
        --data_dir $DATA_DIR \
        --task $TASK \
        --archi $ARCHI \
        --reg_data $REG_DATA \
        --reg_alpha $REG_ALPHA \
        --reg_thresh $REG_THRESH \
        --rgb $RGB \
        --attack $ATTACK \
        --epoch_num $EPOCHS \
        --train_seed $T_SEED \
        --save_dir $MODEL_DIR \
        --train_attack $TRAIN_ATTACK \
        --train_epsilon $TRAIN_EPSILON \
        >> "${MODEL_DIR}/train.log" 2>&1

        cp -r $SAVE_DIR/ $HOME_SAVE_DIR/
        echo "[$SHELL] ## Data copied to scratch"
        echo "[$SHELL] ## Contents of home save dir: $(ls $HOME_SAVE_DIR)"
    fi

    if [ "$RUN_ATTACKS" = true ]; then
        # ATTACK
        echo "Starting attacks for attack seeds $ATTACK_SEEDS"

        for A_SEED in "${ATTACK_SEEDS[@]}"
        do
            for TYPE in "${ATTACK_TYPES[@]}"
            do
                TRIAL_DICT="${MODEL_DIR}/attack_${TYPE}_seed_${A_SEED}"
                mkdir -p "$TRIAL_DICT"

                echo "Running $TYPE attack with attack seed $A_SEED on model $T_SEED"
        
                case $TYPE in
                    "Gaussian")
                        RANGE=$GAUSSIAN_RANGE
                        ;;
                    "Uniform")
                        RANGE="0.0 0.1"
                        ;;
                    "SaltPepper")
                        RANGE="0.0 0.1"
                        ;;
                    "TransferredFGSM")
                        RANGE="0.0 0.3"
                        ;;
                    "BoundaryAttack")
                        RANGE=None
                        ;;
                    *)
                        RANGE="0.0 0.1" # default
                        ;;
                esac

                python "${SCRIPT_DIR}/eval.py" \
                --data_dir $DATA_DIR \
                --task $TASK \
                --archi $ARCHI \
                --reg_data $REG_DATA \
                --reg_alpha $REG_ALPHA \
                --reg_thresh $REG_THRESH \
                --rgb $RGB \
                --attack_seed $A_SEED \
                --epoch_num $EPOCHS \
                --train_seed $T_SEED \
                --save_dir $MODEL_DIR \
                --attack_list "$TYPE" \
                --epsilon_range $RANGE \
                >> "${TRIAL_DICT}/train.log" 2>&1
            done

        done

        cp -r $SAVE_DIR/ $HOME_SAVE_DIR/
        echo "[$SHELL] ## Data copied to scratch"
        echo "[$SHELL] ## Contents of home save dir: $(ls $HOME_SAVE_DIR)"

    fi
done

SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
python "${SCRIPT_DIR}/aggregate_results.py" $GAUSSIAN_RANGE $SAVE_DIR $SEED_STRING > "${HOME_SAVE_DIR}/results_${SEED_STRING}.txt"