#!/bin/bash
# Train Bayesian_LoRA on ARC-Easy (Llama-3.1-8B, seed=42)
set -e

LOG_DIR="/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Easy/Bayesian_LoRA"
TRAIN_LOG="$LOG_DIR/train.log"

mkdir -p $LOG_DIR

cd /home/x-qlan1/rebuttal/Bayesian_lora_peft

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export PYTHONPATH=/home/x-qlan1/rebuttal/Bayesian_lora_peft:$PYTHONPATH
export HF_HOME=/anvil/scratch/x-qlan1/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGING_FACE_HUB_TOKEN=$(cat /home/x-qlan1/.cache/huggingface/token)

echo "[$(date)] === Training Bayesian_LoRA on ARC-Easy ===" | tee "$TRAIN_LOG"

python rebuttal/train_classification.py \
    --method Bayesian_LoRA \
    --dataset ARC-Easy \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir /anvil/scratch/x-qlan1/moule/rebuttal/Bayesian_lora_peft/outputs/Llama3/ARC-Easy/Bayesian_LoRA \
    --target_modules q_proj,k_proj,lm_head \
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
