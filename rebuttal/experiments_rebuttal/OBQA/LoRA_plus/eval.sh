#!/bin/bash
# Evaluate LoRA_plus on OBQA (seed=42, n_mc=1)
set -e

LOG_DIR="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/OBQA/LoRA_plus"
EVAL_LOG="$LOG_DIR/eval.log"

mkdir -p $LOG_DIR

cd /home/anonymous/rebuttal/Bayesian_lora_peft

source /anvil/scratch/anonymous/train-env/bin/activate
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export PYTHONPATH=/home/anonymous/rebuttal/Bayesian_lora_peft:$PYTHONPATH
export HF_HOME=/anvil/scratch/anonymous/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

echo "[$(date)] === Evaluating LoRA_plus on OBQA (n_mc=1) ===" | tee "$EVAL_LOG"

python rebuttal/eval_classification.py \
    --method LoRA_plus \
    --dataset OBQA \
    --model_name_or_path Qwen/Qwen3-14B \
    --lora_path /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/outputs/rebuttal/OBQA/LoRA_plus/final \
    --n_mc 1 \
    --seed 42 \
    --log_dir "$LOG_DIR" \
    --output_file /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/OBQA/LoRA_plus/eval_results.json \
    2>&1 | tee -a "$EVAL_LOG"

echo "[$(date)] === Evaluation finished ===" | tee -a "$EVAL_LOG"
echo "Results:" | tee -a "$EVAL_LOG"
cat /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/OBQA/LoRA_plus/eval_results.json | tee -a "$EVAL_LOG"
