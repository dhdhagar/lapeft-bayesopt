#!/bin/bash
# This script is used to generate and cache features per model for a given word accepted as input

# Define the set of datasets
DATASETS="word2vec-1000 word2vec-2000 word2vec-3000 word2vec-4000"
# Define the set of models
MODELS="t5-base llama-2-7b"  # t5-small t5-large llama-2-13b llama-2-70b
# Define the prompting strategies
PROMPTS="word instruction hint hint-goodness"
# Define the feature aggregation strategies
FEATS="average last-token"
# Define the feature types
FEAT_TYPES="no-additive_features additive_features"
# Get the word from the first argument
TEST_WORDS="computer"
# Define the hint
HINT="Hint: the hidden word is an example of a machine."  # Support for only one hint currently

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Script arguments
        --datasets) DATASETS="$2"; shift ;;
        --test_words) TEST_WORDS="$2"; shift ;;
        --models) MODELS="$2"; shift ;;
        --prompts) PROMPTS="$2"; shift ;;
        --hint) HINT="$2"; shift ;;
        --feats) FEATS="$2"; shift ;;
        --feat_types) FEAT_TYPES="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Iterate over the datasets
for DATASET in $DATASETS; do
    DATA_DIR="data/twentyquestions/datasets/${DATASET}"
    # Iterate over the words
    for WORD in $TEST_WORDS; do
        # Iterate over the models
        for MODEL in $MODELS; do
            # Iterate over the prompting strategies
            for PROMPT in $PROMPTS; do
                # Iterate over the feature aggregation strategies
                for FEAT in $FEATS; do
                    for FEAT_TYPE in $FEAT_TYPES; do
                        echo "
    -----------------------------------
    Generating features --> dataset: '$DATASET', word: '$WORD', model: '$MODEL', prompt: '$PROMPT', hint: '$HINT', features: '$FEAT', feature-type: '$FEAT_TYPE'
    -----------------------------------
    "
                        # Generate features
                        if [[ $MODEL == llama* ]]; then
                            # Don't run first-token aggregation for llama models
                            if [[ $FEAT == "first-token" ]]; then
                                echo "Skipping first-token aggregation for llama models."
                                continue
                            fi
                            # Disable cuda if model is llama (to prevent OOM on blake)
                            CUDA_SETTING="no-cuda"
                        else
                            # Enable cuda if model is t5
                            CUDA_SETTING="cuda"
                        fi
                        # Run extraction
                        python examples/run_fixed_features_20q.py \
                          --data_dir $DATA_DIR \
                          --dataset $WORD \
                          --model $MODEL \
                          --prompt_strategy $PROMPT \
                          --hint "$HINT" \
                          --feat_extraction_strategy $FEAT \
                          --$FEAT_TYPE \
                          --save_word_specific_dataset \
                          --exit_after_feat_extraction \
                          --reset_cache \
                          --$CUDA_SETTING
                    done
                done
            done
        done
    done
done