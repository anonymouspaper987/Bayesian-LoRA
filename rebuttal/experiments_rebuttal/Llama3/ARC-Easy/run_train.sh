#!/bin/bash
# Train all methods on ARC-Easy (Llama-3.1-8B)
set -e

BASE="/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Easy"

for METHOD in LoRA LoRA_plus AdaLoRA Bayesian_LoRA; do
    echo "[$(date)] --- Training $METHOD ---"
    bash "$BASE/$METHOD/train.sh"
done

echo "[$(date)] === All training done ==="
