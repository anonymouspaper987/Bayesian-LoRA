#!/bin/bash
# Evaluate base model (no adapter) on ARC-Easy (Llama-3.1-8B)
set -e

LOG_DIR="/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Easy/base"
EVAL_LOG="$LOG_DIR/eval.log"

mkdir -p $LOG_DIR

cd /home/x-qlan1/rebuttal/Bayesian_lora_peft

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export PYTHONPATH=/home/x-qlan1/rebuttal/Bayesian_lora_peft:$PYTHONPATH
export HF_HOME=/anvil/scratch/x-qlan1/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGING_FACE_HUB_TOKEN=$(cat /home/x-qlan1/.cache/huggingface/token)

echo "[$(date)] === Evaluating base model on ARC-Easy ===" | tee "$EVAL_LOG"

python rebuttal/eval_classification.py \
    --dataset ARC-Easy \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --seed 42 \
    --log_dir "$LOG_DIR" \
    --output_file "$LOG_DIR/eval_results.json" \
    2>&1 | tee -a "$EVAL_LOG"

echo "[$(date)] === Evaluation finished ===" | tee -a "$EVAL_LOG"
