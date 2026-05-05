#!/bin/bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
 
conda activate eat

cd "$(dirname "$0")"

DEBUG=true

NET_TYPE=rgbedge
DATA_DIR=cifar10

if [ "$DEBUG" = true ] ; then
    ROOT="./debug"
    TRAIN_SEEDS=(0)
    ATTACK_SEEDS=(100)
    EPOCHS=1
    echo "--- RUNNING IN DEBUG MODE ---"
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
        --net_type $NET_TYPE \
        --model cifar10 \
        --sigmas 8 \
        --data_dir $DATA_DIR  \
        --classes 10 \
        --epochs $EPOCHS \
        --inp_size 64 \
        --root_dir $ROOT \
        --seed $T_SEED \
        --attack_seeds ${ATTACK_SEEDS[@]}
done

SEED_STRING=$(echo "${TRAIN_SEEDS[*]}" | tr ' ' '_')
python compute_stats.py \
    --root_dir $ROOT \
    --data_dir $DATA_DIR \
    --seeds "${TRAIN_SEEDS[@]}" \
    --out "${ROOT}/Res${DATA_DIR}/seeds_$SEED_STRING.json"