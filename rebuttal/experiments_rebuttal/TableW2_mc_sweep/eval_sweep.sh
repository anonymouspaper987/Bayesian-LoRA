#!/bin/bash
# Table W2: MC Sample Count Sweep on ARC-Challenge (Bayesian-LoRA, seed=42)
# Reuses the trained model from Table W1-A
set -e

LOG_DIR="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableW2_mc_sweep"
SWEEP_LOG="$LOG_DIR/sweep.log"
LORA_PATH="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/outputs/rebuttal/ARC-Easy/Bayesian_LoRA/final"
OUTPUT_BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableW2_mc_sweep"

cd /home/anonymous/rebuttal/Bayesian_lora_peft

source /anvil/scratch/anonymous/train-env/bin/activate
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export PYTHONPATH=/home/anonymous/rebuttal/Bayesian_lora_peft:$PYTHONPATH
export HF_HOME=/anvil/scratch/anonymous/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

echo "[$(date)] === Table W2: MC Sample Count Sweep ===" | tee "$SWEEP_LOG"

for N_MC in 32; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "[$(date)] --- n_mc=$N_MC ---" | tee -a "$SWEEP_LOG"

    python rebuttal/eval_classification.py \
        --method Bayesian_LoRA \
        --dataset ARC-Easy \
        --model_name_or_path Qwen/Qwen3-14B \
        --lora_path "$LORA_PATH" \
        --n_mc $N_MC \
        --seed 42 \
        --output_file "$OUTPUT_BASE/eval_nmc${N_MC}.json" \
        2>&1 | tee -a "$SWEEP_LOG"

    echo "Results (n_mc=$N_MC):" | tee -a "$SWEEP_LOG"
    cat "$OUTPUT_BASE/eval_nmc${N_MC}.json" | tee -a "$SWEEP_LOG"
    echo "" | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "[$(date)] === Sweep finished ===" | tee -a "$SWEEP_LOG"
