#!/bin/bash -e

# Job defaults
desc="20q"
partition="gpu"
n_gpus="1080ti:1"
mem="100G"
time="0-2:00:00"

# Script defaults
MODELS="t5-small t5-base t5-large llama-2-7b llama-2-13b"  # llama-2-70b
PROMPTS="word instruction hint hint-goodness"
HINT="Hint: the hidden word is an example of a machine."  # Support for only one hint currently
FEATS="average last-token"
TEST_WORDS="computer"
N_INIT_DATAS="5"  # "1 5 10"
N_SEEDS=5
STEPS="100"  # "50 100"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Job arguments
        --partition) partition="$2"; shift ;;
        --n_gpus) n_gpus="$2"; shift ;;
        --mem) mem="$2"; shift ;;
        --time) time="$2"; shift ;;
        # Script arguments
        --test_words) TEST_WORDS="$2"; shift ;;
        --n_init_datas) N_INIT_DATAS="$2"; shift ;;
        --n_seeds) N_SEEDS="$2"; shift ;;
        --models) MODELS="$2"; shift ;;
        --prompts) PROMPTS="$2"; shift ;;
        --hint) HINT="$2"; shift ;;
        --feats) FEATS="$2"; shift ;;
        --steps) STEPS="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

for TEST_WORD in $TEST_WORDS; do
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
                for N_INIT_DATA in $N_INIT_DATAS; do
                    for STEP in $STEPS; do
                        ./scripts/run_20q.sh \
                            --test_word $TEST_WORD \
                            --n_init_data $N_INIT_DATA \
                            --n_seeds $N_SEEDS \
                            --model $MODEL \
                            --prompt $PROMPT \
                            --hint "$HINT" \
                            --feat $FEAT \
                            --steps $STEP \
                            --partition $partition \
                            --n_gpus $n_gpus \
                            --mem $mem \
                            --time $time
                    done
                done
            done
        done
    done
done
