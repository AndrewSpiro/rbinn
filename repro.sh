#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate pixelreg

DEBUG=false
RUN_TRAINING=true

TASK=CIFAR10
ARCHI=ResNet18
REG_DATA=CIFAR10
REG_ALPHA=10
REG_THRESH=0.8
RGB=true
ATTACK=false
ATTACK_TYPES=(Gaussian)

if [ "$DEBUG" = true ]; then
    EPOCHS=1
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    echo "--- RUNNING IN DEBUG MODE ---"
else
    EPOCHS=40
    TRAIN_SEEDS=(2)
    ATTACK_SEEDS=(100 101 102)
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do

    MODEL_DIR="save/model_seed_$T_SEED"
    mkdir -p "$MODEL_DIR"

    if [ "$RUN_TRAINING" = true ]; then
        #Train
        echo "Starting training for train seed $T_SEED"
        python train.py \
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
        >> "${MODEL_DIR}/train.log" 2>&1
    fi

    # ATTACK
    for A_SEED in "${ATTACK_SEEDS[@]}"
    do
        for TYPE in "${ATTACK_TYPES[@]}"
        do
            TRIAL_DICT="${MODEL_DIR}/attack_${TYPE}_seed_${A_SEED}"
            mkdir -p "$TRIAL_DICT"

            echo "Running $TYPE attack with attack seed $A_SEED on model $T_SEED"
    
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
                    RANGE="0.0 0.1" # default
                    ;;
            esac

            python eval.py \
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
done

SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
python aggregate_results.py > "save/results_${SEED_STRING}.txt"