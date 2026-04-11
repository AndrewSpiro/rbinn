#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate pixelreg

DEBUG=true

if [ "$DEBUG" = true ]; then
    EPOCHS=1
    TRAIN_SEEDS=(0 1)
    ATTACK_SEEDS=(100 101)
    echo "--- RUNNING IN DEBUG MODE ---"
else
    EPOCHS=40
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
fi

ATTACK_TYPES=(Gaussian)

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    echo "Starting training for train seed $T_SEED"

    MODEL_DIR="save/model_seed_$T_SEED"
    mkdir -p "$MODEL_DIR"

    #Train
    python train.py --epoch_num $EPOCHS --attack False --train_seed $T_SEED --save_dir $MODEL_DIR >> "${MODEL_DIR}/train.log" 2>&1

    # ATTACK
    for A_SEED in "${ATTACK_SEEDS[@]}"
    do
        for TYPE in "${ATTACK_TYPES[@]}"
        do
            TRIAL_DICT="${MODEL_DIR}/attack_${TYPE}_seed_${A_SEED}"
            mkdir -p "$TRIAL_DICT"

            echo "Running $TYPE attack with attack seed $A_SEED on model T_SEED"
    
            case $TYPE in
                "Gaussian")
                    RANGE="0.0 0.3"
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
                    RANGE="0.0 0.1" # Default
                    ;;
            esac

            python eval.py \
            --attack_seed $A_SEED \
            --epoch_num $EPOCHS \
            --save_dir $MODEL_DIR \
            --attack_list "$TYPE" \
            --epsilon_range $RANGE \
            >> "${MODEL_DIR}/train.log" 2>&1
        done

    done
done