#!/bin/bash -e

# Define a function to split a word by spaces and join by hyphen
split_and_join() {
  local word="$1"
  local hyphenated_word=""

  # Split the word by spaces
  while IFS= read -r -d ' ' part || [[ -n "$part" ]]; do
    # Append each part with a hyphen
    hyphenated_word+="${part}-"
  done <<< "$word"

  # Remove the trailing hyphen
  hyphenated_word="${hyphenated_word%-}"

  echo "$hyphenated_word"
}

# Job defaults
desc="20q"
partition="gpu"
n_gpus="1080ti:1"
mem="100G"
time="0-1:00:00"

# Script defaults
MODEL="t5-small"
PROMPT="word"
HINT=""
FEAT="average"
TEST_WORD="computer"
N_INIT_DATA=5
N_SEEDS=5
STEPS=100

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
        --hint) HINT="$2"; shift ;;
        --feat) FEAT="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Echo all job arguments in one line
echo "
Job arguments: desc=${desc}, partition=${partition}, n_gpus=${n_gpus}, mem=${mem}, time=${time}"
# Echo all script arguments in one line
echo "Script arguments: test_word=${TEST_WORD}, n_init_data=${N_INIT_DATA}, n_seeds=${N_SEEDS}, model=${MODEL}, prompt=${PROMPT}, hint="${HINT}", feat=${FEAT}, steps=${STEPS}"

# Create log directory for the job
job_dir="jobs/${desc}"
mkdir -p ${job_dir}

# Determine output dir for script
if [[ $PROMPT == "hint" ]]; then
  EXPERIMENT="${TEST_WORD}_${MODEL}_${PROMPT}-$(split_and_join "${HINT}")_${FEAT}_n${N_INIT_DATA}_t${STEPS}"
else
  EXPERIMENT="${TEST_WORD}_${MODEL}_${PROMPT}_${FEAT}_n${N_INIT_DATA}_t${STEPS}"
fi
OUT_DIR="outputs/${desc}/${EXPERIMENT}"

# Set RUN_ID to the current timestamp
RUN_ID="$(date +%s)"

# Submit job
JOB_DESC=${desc}_${EXPERIMENT} && JOB_NAME=${JOB_DESC}_${RUN_ID} && \
  sbatch -J ${JOB_NAME} -e ${job_dir}/${JOB_NAME}.err -o ${job_dir}/${JOB_NAME}.log \
    --partition=${partition} --gres=gpu:${n_gpus} --mem=${mem} --time=${time} scripts/run_sbatch.sh \
      examples/run_fixed_features_20q.py \
      --run_id="${RUN_ID}" \
      --dataset="${TEST_WORD}" \
      --n_init_data=${N_INIT_DATA} \
      --n_seeds=${N_SEEDS} \
      --model="${MODEL}" \
      --prompt_strategy="${PROMPT}" \
      --hint="${HINT}" \
      --feat_extraction_strategy="${FEAT}" \
      --T=${STEPS} \
      --out_dir="${OUT_DIR}"
echo "Log path: ${job_dir}/${JOB_NAME}.log"
echo "Output path: ${OUT_DIR}/${RUN_ID}
"
