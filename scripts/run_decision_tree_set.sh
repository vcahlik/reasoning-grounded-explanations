#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

MODELS=(
    # "unsloth/llama-3-8b-Instruct-bnb-4bit"
    # "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    # "unsloth/zephyr-sft-bnb-4bit"
    # "unsloth/phi-4-bnb-4bit"
    # "unsloth/gemma-2-9b-it-bnb-4bit"
    # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)

DEPTHS=(
    # "4"
    "5"
    # "6"
    "7"
    # "8"
    "9"
    # "10"
)

RESPONSE_MODES=(
    # "decision"
    # "explanation"
    "decision explanation"
    "decision explanation reasoning"
)
# OTHER_FLAGS="--in-context-learning --in-context-learning-examples 20"
OTHER_FLAGS=""

for MODEL in "${MODELS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for RESPONSE_MODE in "${RESPONSE_MODES[@]}"; do
            python "$SCRIPT_DIR/run_experiment.py" \
                --model-name "$MODEL" \
                --problem-type "decision_tree" \
                --dataset-base-path "data/datasets/decision_trees_v10/n_inputs_2_depth_$DEPTH" \
                --dataset-id 0 \
                --response-modes $RESPONSE_MODE \
                --train-set-size 2000 \
                --test-set-size 200 \
                --batch-size 4 \
                --learning-rate "0.00005" \
                --max-new-tokens 1000 \
                --output-dir-path-prefix "outputs/exp" \
                $OTHER_FLAGS
        done
    done
done
