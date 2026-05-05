#!/bin/bash

set -e

source $(conda info --base)/etc/profile.d/conda.sh

DEBUG=true
RUN_VALIDS=false

MODELS=("pixelreg" "eat" "cnnf" "vonenet")
EPSILON_SPACE=berger
PGD_NUM_ITER=40
PGD_STEP_SIZE=0.01

echo "Starting full pipeline..."

if [ "$RUN_VALIDS" = true ]; then
    echo "Starting validations..."
    echo "Starting BINN1 pipeline..."
    bash binn1_pixelreg/repro.sh
else
    echo "Skipping validations"
fi

conda activate verona_env
for m in "${MODELS[@]}"; do
    echo "Obtaining robustness distribution for $m"
    python create_robustness_dist.py $m \
        --epsilon_space $EPSILON_SPACE \
        --bool_debug $DEBUG \
        --pgd_num_iter $PGD_NUM_ITER \
        --pgd_step_size $PGD_STEP_SIZE
done

echo "Completed all experiments successfully"