#!/bin/bash

set -e

source $(conda info --base)/etc/profile.d/conda.sh

DEBUG=true
RUN_VALIDS=false
GET_RDS=true
AGG_RESULTS=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
echo "[$SHELL] ## Data dir is here: ${DATA_DIR}"
mkdir -p "$DATA_DIR"
EXP_REPO_PATH="${SCRIPT_DIR}/experiments"

declare -A MODEL_IDS=( ["pixelreg"]=1 ["eat"]=3 ["cnnf"]=4 ["vonenet"]=5)
MODELS=("eat")
EPSILON_SPACE=berger
PGD_NUM_ITER=40
PGD_STEP_SIZE=0.01

echo "[$SHELL] ## Starting full pipeline..."

if [ "$RUN_VALIDS" = true ]; then
    echo "[$SHELL] ## Starting validations..."

    for m in "${MODELS[@]}"
    do
        echo "[$SHELL] ## Starting $m pipeline..."
    
        conda deactivate
        conda activate $m

        bash "${SCRIPT_DIR}/binn${MODEL_IDS[$m]}_"${m}"/repro.sh"

        echo "[$SHELL] ## Completed $m validation."
    done
else
    echo "[$SHELL] ## Skipping validations"
fi

conda activate verona_env

if [ "${GET_RDS}" = true ]; then
    for m in "${MODELS[@]}"; do
        echo "[$SHELL] ## Obtaining robustness distribution for $m"
        python create_robustness_dist.py $m \
            --epsilon_space $EPSILON_SPACE \
            --bool_debug $DEBUG \
            --pgd_num_iter $PGD_NUM_ITER \
            --pgd_step_size $PGD_STEP_SIZE \
            --data_dir $DATA_DIR \
            --exp_repo_path "${EXP_REPO_PATH}" \
            --model_seed 0 \
            --train_type clean
    done
fi

if [ "${AGG_RESULTS}" = true ]; then
    echo "[$SHELL] ## WARNING: Aggregating results. Please manually update your experiments json with the desired experiments to include. ####"

    if [ "${DEBUG}" = true ]; then
        RESULTS_DIR=${SCRIPT_DIR}/debug_rd_results
    else
        RESULTS_DIR=${SCRIPT_DIR}/rd_results
    fi
    mkdir -p $RESULTS_DIR

    python rd_analysis.py --experiments_path experiments.json --results_dir "${RESULTS_DIR}" 
fi

echo "[$SHELL] ## Completed all experiments successfully"