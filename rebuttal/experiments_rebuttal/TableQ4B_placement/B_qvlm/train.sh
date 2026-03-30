#!/bin/bash
# Table Q4-B: Adapter placement B (LoRA-style) - q_proj,v_proj,lm_head
set -e

LOG_DIR="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/B_qvlm"
TRAIN_LOG="$LOG_DIR/train.log"

mkdir -p $LOG_DIR

cd /home/anonymous/rebuttal/Bayesian_lora_peft

source /anvil/scratch/anonymous/train-env/bin/activate
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export PYTHONPATH=/home/anonymous/rebuttal/Bayesian_lora_peft:$PYTHONPATH
export HF_HOME=/anvil/scratch/anonymous/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

echo "[$(date)] === Training Bayesian_LoRA placement B ===" | tee "$TRAIN_LOG"

python rebuttal/train_classification.py \
    --method Bayesian_LoRA \
    --dataset ARC-Easy \
    --model_name_or_path Qwen/Qwen3-14B \
    --output_dir /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/outputs/rebuttal/TableQ4B/B_qvlm \
    --target_modules q_proj,v_proj,lm_head \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --max_length 512 \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --log_dir "$LOG_DIR" \
    2>&1 | tee -a "$TRAIN_LOG"

echo "[$(date)] === Training finished ===" | tee -a "$TRAIN_LOG"
