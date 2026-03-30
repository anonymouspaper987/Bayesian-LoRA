#!/bin/bash
# Run all ARC-Easy evals (all methods, includes PRR metric)
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/ARC-Easy"

echo "[$(date)] === Starting ARC-Easy eval (all methods) ==="

for METHOD in base LoRA LoRA_plus AdaLoRA Bayesian_LoRA; do
    echo "[$(date)] --- $METHOD ---"
    bash "$BASE/$METHOD/eval.sh"
done

echo "[$(date)] === All done ==="
