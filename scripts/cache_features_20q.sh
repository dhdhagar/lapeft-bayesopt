#!/bin/bash
# This script is used to generate and cache features per model for a given word accepted as input

# Get the word from the first argument
WORD="computer"
# Define the set of models
MODELS="t5-small t5-base t5-large llama-2-7b llama-2-13b llama-2-70b"
# Define the prompting strategies
PROMPTS="word instruction"  # "hint hint-goodness"
HINT=""  # Support for only one hint currently
# Define the feature aggregation strategies
FEATS="average last-token"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Script arguments
        --test_words) TEST_WORDS="$2"; shift ;;
        --models) MODELS="$2"; shift ;;
        --prompts) PROMPTS="$2"; shift ;;
        --hint) HINT="$2"; shift ;;
        --feats) FEATS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Iterate over the words
for WORD in $TEST_WORDS; do
    # Iterate over the models
    for MODEL in $MODELS; do
        # Iterate over the prompting strategies
        for PROMPT in $PROMPTS; do
            # Iterate over the feature aggregation strategies
            for FEAT in $FEATS; do
                echo "
-----------------------------------
Generating features for word: '$WORD', model: '$MODEL', prompt: '$PROMPT', hint: '$HINT', aggregation: '$AGGREGATION'
-----------------------------------
"
                # Generate the features
                if [[ $MODEL == llama* ]]; then
                    # Don't run if aggregation is first-token for llama models
                    if [[ $FEAT == "first-token" ]]; then
                        echo "Skipping first-token aggregation for llama models."
                        continue
                    fi

                    # Run without cuda if model is llama (doing this to prevent OOM on blake)
                    python examples/run_fixed_features_20q.py \
                      --dataset $WORD \
                      --model $MODEL \
                      --prompt_strategy $PROMPT \
                      --hint "$HINT" \
                      --feat_extraction_strategy $FEAT \
                      --save_word_specific_dataset \
                      --exit_after_feat_extraction \
                      --reset_cache \
                      --no-cuda
                else
                    # Run with cuda if model is t5
                    python examples/run_fixed_features_20q.py \
                      --dataset $WORD \
                      --model $MODEL \
                      --prompt_strategy $PROMPT \
                      --hint "$HINT" \
                      --feat_extraction_strategy $FEAT \
                      --save_word_specific_dataset \
                      --exit_after_feat_extraction \
                      --reset_cache
                fi
            done
        done
    done
done
