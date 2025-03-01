#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

MODELS=(
    "unsloth/llama-3-8b-Instruct-bnb-4bit"
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    "unsloth/zephyr-sft-bnb-4bit"
    "unsloth/phi-4-bnb-4bit"
    # "unsloth/gemma-2-9b-it-bnb-4bit"
    # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)

ALL_N_INPUTS=(
    "8"
    # "10"
    # "12"
    # "14"
    # "16"
)

RESPONSE_MODES=(
    "decision"
    "explanation"
    # "decision explanation"
    # "decision explanation reasoning"
)
OTHER_FLAGS="--in-context-learning --in-context-learning-examples 5"
# OTHER_FLAGS=""

for MODEL in "${MODELS[@]}"; do
    for N_INPUTS in "${ALL_N_INPUTS[@]}"; do
        for RESPONSE_MODE in "${RESPONSE_MODES[@]}"; do
            python "$SCRIPT_DIR/run_experiment.py" \
                --model-name "$MODEL" \
                --problem-type logistic_regressor \
                --dataset-base-path "data/datasets/logistic_regressors_v2/n_inputs_$N_INPUTS" \
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
