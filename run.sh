#!/bin/bash

set -e

source $(conda info --base)/etc/profile.d/conda.sh

DEBUG=true
RUN_VALIDS=false
GET_RDS=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
echo "Data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"
EXP_REPO_PATH="${SCRIPT_DIR}/experiments"

declare -A MODEL_IDS=( ["pixelreg"]=1 ["eat"]=3 ["cnnf"]=4 ["vonenet"]=5)
MODELS=("eat")
EPSILON_SPACE=berger
PGD_NUM_ITER=40
PGD_STEP_SIZE=0.01

echo "Starting full pipeline..."

if [ "$RUN_VALIDS" = true ]; then
    echo "Starting validations..."

    for m in "${MODELS[@]}"
    do
        echo "Starting $m pipeline..."
    
        conda deactivate
        conda activate $m

        bash "${SCRIPT_DIR}/binn${MODEL_IDS[$m]}_"${m}"/repro.sh"

        echo "Completed $m validation."
    done
else
    echo "Skipping validations"
fi

if [ "${GET_RDS}" = true ]; then
    conda activate verona_env
    for m in "${MODELS[@]}"; do
        echo "Obtaining robustness distribution for $m"
        python create_robustness_dist.py $m \
            --epsilon_space $EPSILON_SPACE \
            --bool_debug $DEBUG \
            --pgd_num_iter $PGD_NUM_ITER \
            --pgd_step_size $PGD_STEP_SIZE \
            --data_dir $DATA_DIR \
            --exp_repo_path "${EXP_REPO_PATH}"
    done
fi
echo "Completed all experiments successfully"