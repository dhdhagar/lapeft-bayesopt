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
partition="gpu"  # gypsum-1080ti on Unity
constraint=""
n_gpus="1"  # 1080ti:1 on Blake
mem="20G"
n_cpus="1"
time="0-6:00:00"

# Script defaults
DATASET="word2vec-2000"  # word2vec-1000 word2vec-2000 word2vec-3000 word2vec-4000
MODEL="t5-base"
PROMPT="word"
HINT="Hint: the hidden word is an example of a machine."
FEAT="average"
FEAT_TYPE="no-additive_features"
TEST_WORD="computer"
SURROGATE="laplace"
ACQUISITION="thompson_sampling"
N_INIT_DATA=5
N_SEEDS=5
STEPS=100
WILDCARD=""
OUTPUTS="outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Job arguments
        --desc) desc="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --constraint) constraint="$2"; shift ;;
        --n_gpus) n_gpus="$2"; shift ;;
        --n_cpus) n_cpus="$2"; shift ;;
        --mem) mem="$2"; shift ;;
        --time) time="$2"; shift ;;
        # Script arguments
        --dataset) DATASET="$2"; shift ;;
        --test_word) TEST_WORD="$2"; shift ;;
        --n_init_data) N_INIT_DATA="$2"; shift ;;
        --n_seeds) N_SEEDS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --prompt) PROMPT="$2"; shift ;;
        --hint) HINT="$2"; shift ;;
        --feat) FEAT="$2"; shift ;;
        --feat_type) FEAT_TYPE="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        --surrogate) SURROGATE="$2"; shift ;;
        --acquisition) ACQUISITION="$2"; shift ;;
        --wildcard) WILDCARD="$2"; shift ;;
        --outputs) OUTPUTS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Echo all job arguments in one line
echo "
Job arguments: desc=${desc}, partition=${partition}, n_gpus=${n_gpus}, mem=${mem}, time=${time}"
# Echo all script arguments in one line
echo "Script arguments: dataset=${DATASET}, test_word=${TEST_WORD}, n_init_data=${N_INIT_DATA}, n_seeds=${N_SEEDS}, \
model=${MODEL}, prompt=${PROMPT}, hint="${HINT}", feat=${FEAT}, feat_type=${FEAT_TYPE}, steps=${STEPS}, \
surrogate=${SURROGATE}, acquisition=${ACQUISITION}${WILDCARD:+", wildcard=${WILDCARD}"}"

# Create log directory for the job
job_dir="jobs/${desc}"
mkdir -p ${job_dir}

# If additive, then set label as "additive", else leave it blank
if [[ $FEAT_TYPE == "additive_features" ]]; then
  FEAT_LABEL="_additive"
else
  FEAT_LABEL=""
fi

# Determine output dir for script
if [[ $PROMPT == hint* ]]; then
  HINT_LABEL="-$(split_and_join "${HINT}")"
else
  HINT_LABEL=""
fi
# Create WILDCARD_LABEL
WILDCARD_LABEL=""
if [[ -n $WILDCARD ]]; then
  WILDCARD_LABEL="_$(split_and_join "${WILDCARD}")"
fi
EXPERIMENT="${TEST_WORD}_${SURROGATE}_${ACQUISITION}_${MODEL}_${PROMPT}${HINT_LABEL}_${FEAT}${FEAT_LABEL}\
_n${N_INIT_DATA}_t${STEPS}${WILDCARD_LABEL}"
OUT_DIR="${OUTPUTS}/${desc}/${DATASET}/${EXPERIMENT}"

# Determine data_dir
DATA_DIR="data/twentyquestions/datasets/${DATASET}"

# Set RUN_ID to the current timestamp
RUN_ID="$(date +%s)"

# Submit job
JOB_DESC=${desc}_${EXPERIMENT} && JOB_NAME=${JOB_DESC}_${RUN_ID} && \
  sbatch -J ${JOB_NAME} -e ${job_dir}/${JOB_NAME}.err -o ${job_dir}/${JOB_NAME}.log \
    --partition="${partition}"${constraint:+ --constraint "${constraint}"} --gres=gpu:${n_gpus} --cpus-per-task=${n_cpus} \
    --mem=${mem} --time=${time} \
    scripts/run_sbatch.sh examples/run_fixed_features_20q.py \
      --run_id="${RUN_ID}" \
      --data_dir="${DATA_DIR}" \
      --dataset="${TEST_WORD}" \
      --n_init_data=${N_INIT_DATA} \
      --n_seeds=${N_SEEDS} \
      --model="${MODEL}" \
      --prompt_strategy="${PROMPT}" \
      --hint="${HINT}" \
      --feat_extraction_strategy="${FEAT}" \
      --${FEAT_TYPE} \
      --surrogate_fn="${SURROGATE}" \
      --acquisition_fn="${ACQUISITION}" \
      --T=${STEPS} \
      --out_dir="${OUT_DIR}"${WILDCARD:+ ${WILDCARD}}

echo "Log path: ${job_dir}/${JOB_NAME}.log"
echo "Output path: ${OUT_DIR}/${RUN_ID}
"
