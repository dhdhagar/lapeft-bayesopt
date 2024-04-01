#!/bin/bash
# This script is used to generate and cache features per model for a given word accepted as input

# Get the word from the first argument
WORD=$1
# Define the set of models
MODELS="t5-small t5-base t5-large llama-2-7b llama-2-13b llama-2-70b"
# Define the prompting strategies
PROMPTS="word instruction"
# Define the feature aggregation strategies
AGGREGATIONS="average last-token"

# Iterate over the models
for MODEL in $MODELS; do
    # Iterate over the prompting strategies
    for PROMPT in $PROMPTS; do
        # Iterate over the feature aggregation strategies
        for AGGREGATION in $AGGREGATIONS; do
            echo "Generating features for word: $WORD, model: $MODEL, prompt: $PROMPT, aggregation: $AGGREGATION"
            echo "-----------------------------------"
            # Generate the features
            if [[ $MODEL == llama* ]]; then
                # Run without cuda if model is llama (doing this to prevent OOM on blake)
                python examples/run_fixed_features_20q.py \
                  --test_idx_or_word $WORD \
                  --model $MODEL \
                  --prompt_strategy $PROMPT \
                  --feat_extraction_strategy $AGGREGATION \
                  --save_word_specific_dataset \
                  --exit_after_feat_extraction \
                  --no-cuda
            else
                # Run with cuda if model is t5
                python examples/run_fixed_features_20q.py \
                  --test_idx_or_word $WORD \
                  --model $MODEL \
                  --prompt_strategy $PROMPT \
                  --feat_extraction_strategy $AGGREGATION \
                  --save_word_specific_dataset \
                  --exit_after_feat_extraction
            fi
        done
    done
done
