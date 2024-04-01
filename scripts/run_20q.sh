#!/bin/bash -e

# Job defaults
desc="20q"
partition="gpu"
n_gpus=1
mem="40G"
time="0-1:00:00"

# Script defaults
MODEL="t5-small"
PROMPT="word"
FEAT="average"
TEST_WORD="computer"
N_INIT_DATA=10
N_SEEDS=5
STEPS=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Job arguments
        --desc) desc="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --n_gpus) n_gpus="$2"; shift ;;
        --mem) mem="$2"; shift ;;
        --time) time="$2"; shift ;;
        # Script arguments
        --test_word) TEST_WORD="$2"; shift ;;
        --n_init_data) N_INIT_DATA="$2"; shift ;;
        --n_seeds) N_SEEDS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --prompt) PROMPT="$2"; shift ;;
        --feat) FEAT="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Echo all job arguments in one line
echo "Job arguments: desc=${desc}, partition=${partition}, n_gpus=${n_gpus}, mem=${mem}, time=${time}"
# Echo all script arguments in one line
echo "Script arguments: test_word=${TEST_WORD}, n_init_data=${N_INIT_DATA}, n_seeds=${N_SEEDS}, model=${MODEL}, prompt=${PROMPT}, feat=${FEAT}, steps=${STEPS}"

# Create log directory for the job
job_dir="jobs/${desc}"
mkdir -p ${job_dir}

# Determine output dir for script
OUT_DIR="outputs/${desc}/${TEST_WORD}_${MODEL}_${PROMPT}_${FEAT}_${N_INIT_DATA}_${STEPS}"

# Submit job
JOB_DESC=${desc} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e ${job_dir}/${JOB_NAME}.err -o ${job_dir}/${JOB_NAME}.log \
    --partition=${partition} --gres=gpu:${n_gpus} --mem=${mem} --time=${time} scripts/run_sbatch.sh \
      examples/run_fixed_features_20q.py \
      --test_idx_or_word="${TEST_WORD}" \
      --n_init_data=${N_INIT_DATA} \
      --n_seeds=${N_SEEDS} \
      --model="${MODEL}" \
      --prompt_strategy="${PROMPT}" \
      --feat_extraction_strategy="${FEAT}" \
      --T=${STEPS} \
      --out_dir=${OUT_DIR}
echo "Logs path: ${job_dir}/${JOB_NAME}"
echo "Output path: ${out_dir}
"
