#!/bin/bash

DEBUG=true

if [ "$DEBUG" = true ] ; then
    ROOT="./debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    EPOCHS=1
else
    ROOT="."
    TRAIN_SEEDS=(0 1 2)
    ATTACK_SEEDS=(100 101 102)
    EPOCHS=10
fi

for T_SEED in "${TRAIN_SEEDS[@]}"
do
    echo "Running training with seed $T_SEED"

    python train.py \
        --net_type rgbedge \
        --model cifar10 \
        --sigmas 8 \
        --data_dir cifar10  \
        --classes 10 \
        --epochs $EPOCHS \
        --inp_size 28 \
        --root_dir $ROOT \
        --seed $T_SEED \
        --attack_seeds ${ATTACK_SEEDS[@]}
done

