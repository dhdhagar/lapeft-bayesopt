#!/bin/bash -e

# Job defaults
desc="20q"
partition="gpu"
n_gpus="1080ti:1"
mem="100G"
n_cpus="4"
time="0-6:00:00"

# Script defaults
DATASETS="word2vec-2000"  # word2vec-1000 word2vec-2000 word2vec-3000 word2vec-4000
MODELS="t5-base llama-2-7b"  # t5-small t5-base t5-large llama-2-7b llama-2-13b llama-2-70b
PROMPTS="word instruction hint hint-goodness"
FEATS="average last-token"
FEAT_TYPES="no-additive_features additive_features"
N_INIT_DATAS="5"  # "1 5 10"
N_SEEDS=5
STEPS="100"  # "50 100"
TEST_WORDS="computer"
HINT="Hint: the hidden word is an example of a machine."  # Support for only one hint currently
SURROGATES="laplace gp"
ACQUISITIONS="thompson_sampling logEI"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Job arguments
        --partition) partition="$2"; shift ;;
        --n_gpus) n_gpus="$2"; shift ;;
        --n_cpus) n_cpus="$2"; shift ;;
        --mem) mem="$2"; shift ;;
        --time) time="$2"; shift ;;
        # Script arguments
        --datasets) DATASETS="$2"; shift ;;
        --test_words) TEST_WORDS="$2"; shift ;;
        --n_init_datas) N_INIT_DATAS="$2"; shift ;;
        --n_seeds) N_SEEDS="$2"; shift ;;
        --models) MODELS="$2"; shift ;;
        --prompts) PROMPTS="$2"; shift ;;
        --hint) HINT="$2"; shift ;;
        --feats) FEATS="$2"; shift ;;
        --feat_types) FEAT_TYPES="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        --surrogates) SURROGATES="$2"; shift ;;
        --acquisitions) ACQUISITIONS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

for DATASET in $DATASETS; do
    for TEST_WORD in $TEST_WORDS; do
        for SURROGATE in $SURROGATES; do
            for ACQUISITION in $ACQUISITIONS; do
                for MODEL in $MODELS; do
                    for PROMPT in $PROMPTS; do
                        for FEAT in $FEATS; do
                            # Don't run if aggregation is first-token for llama models
                            if [[ $MODEL == llama* ]]; then
                                if [[ $FEAT == "first-token" ]]; then
                                    echo "Skipping first-token feature extraction for llama models."
                                    continue
                                fi
                            fi
                            for FEAT_TYPE in $FEAT_TYPES; do
                                # Skip if feat type is additive and prompt is word
                                if [[ $FEAT_TYPE == "additive_features" && $PROMPT == "word" ]]; then
                                    echo "Skipping additive features with word prompt."
                                    continue
                                fi
                                for N_INIT_DATA in $N_INIT_DATAS; do
                                    for STEP in $STEPS; do
                                        ./scripts/run_20q.sh \
                                            --dataset $DATASET \
                                            --test_word $TEST_WORD \
                                            --n_init_data $N_INIT_DATA \
                                            --n_seeds $N_SEEDS \
                                            --model $MODEL \
                                            --prompt $PROMPT \
                                            --hint "$HINT" \
                                            --feat $FEAT \
                                            --feat_type $FEAT_TYPE \
                                            --steps $STEP \
                                            --partition $partition \
                                            --n_gpus $n_gpus \
                                            --n_cpus $n_cpus \
                                            --mem $mem \
                                            --time $time \
                                            --surrogate $SURROGATE \
                                            --acquisition $ACQUISITION
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done