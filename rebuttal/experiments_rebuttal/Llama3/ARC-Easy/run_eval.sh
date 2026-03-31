#!/bin/bash
# Evaluate all methods on ARC-Easy (Llama-3.1-8B)
set -e

BASE="/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Easy"

for METHOD in base LoRA LoRA_plus AdaLoRA Bayesian_LoRA; do
    echo "[$(date)] --- Evaluating $METHOD ---"
    bash "$BASE/$METHOD/eval.sh"
done

echo "[$(date)] === All evaluation done ==="
