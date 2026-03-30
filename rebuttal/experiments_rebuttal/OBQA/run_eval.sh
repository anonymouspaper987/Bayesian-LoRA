#!/bin/bash
# Run all OBQA evals (all methods, includes PRR metric)
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/OBQA"

echo "[$(date)] === Starting OBQA eval (all methods) ==="

for METHOD in LoRA LoRA_plus AdaLoRA Bayesian_LoRA; do
    echo "[$(date)] --- $METHOD ---"
    bash "$BASE/$METHOD/eval.sh"
done

echo "[$(date)] === All done ==="
