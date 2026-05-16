#!/bin/bash

#SBATCH --job-name=rd_b3
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s4015355@vuw.leidenuni.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="testing"
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem=2G

set -e

source $(conda info --base)/etc/profile.d/conda.sh

DEBUG=true
RUN_VALIDS=false
GET_RDS=true
AGG_RESULTS=false

# Home directories
SCRIPT_DIR=$PWD
HOME_DATA_DIR="${SCRIPT_DIR}/data"
echo "[$SHELL] ## Home data dir is here: ${HOME_DATA_DIR}"
mkdir -p "$HOME_DATA_DIR"
HOME_EXP_REPO_PATH="${SCRIPT_DIR}/experiments"
echo "[$SHELL] ## Home experiment repository is here: ${HOME_EXP_REPO_PATH}"
mkdir -p "$HOME_EXP_REPO_PATH"

# Scratch directories for fast loading
SCRATCH_PATH="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
SCRATCH_DATA_DIR="${SCRATCH_PATH}/data"
SCRATCH_EXP_REPO_PATH="${SCRATCH_PATH}/experiments"
# Copy data from home to scratch
cp -r $HOME_DATA_DIR $SCRATCH_DATA_DIR
echo "[$SHELL] ## Data copied to scratch"
echo "Contents of scratch data dir: $(ls $SCRATCH_DATA_DIR)"

#### Experiment pipeline ####
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [ "${GET_RDS}" = true ]; then
    for m in "${MODELS[@]}"; do
        echo "[$SHELL] ## Obtaining robustness distribution for $m"
        python create_robustness_dist.py $m \
            --epsilon_space $EPSILON_SPACE \
            --bool_debug $DEBUG \
            --pgd_num_iter $PGD_NUM_ITER \
            --pgd_step_size $PGD_STEP_SIZE \
            --data_dir $SCRATCH_DATA_DIR \
            --exp_repo_path $SCRATCH_EXP_REPO_PATH \
            --model_seed 0 \
            --train_type clean  &
        PID=$!

        # Every 60s, log what the process is waiting on
        while kill -0 $PID 2>/dev/null; do
            sleep 10
            echo "--- $(date) ---" >> strace_$SLURM_JOB_ID.log
            strace -p $PID -e trace=network,file -c 2>> strace_$SLURM_JOB_ID.log || true
        done
        wait $PID
    done
else
    echo "[$SHELL] ## Skipping obtaining RDs"
fi

# Copy experiment results from scratch to home
cp -r $SCRATCH_EXP_REPO_PATH $HOME_EXP_REPO_PATH

if [ "${AGG_RESULTS}" = true ]; then
    echo "[$SHELL] #### Aggregating results. Please manually update your experiments json with the desired experiments to include. ####"

    if [ "${DEBUG}" = true ]; then
        RESULTS_DIR=${SCRIPT_DIR}/debug_rd_results
    else
        RESULTS_DIR=${SCRIPT_DIR}/rd_results
    fi
    mkdir -p $RESULTS_DIR

    python rd_analysis.py --experiments_path experiments.json --results_dir "${RESULTS_DIR}"
else
    echo "[$SHELL] ## Skipping aggregating results"
fi

echo "[$SHELL] ## Completed all experiments successfully"