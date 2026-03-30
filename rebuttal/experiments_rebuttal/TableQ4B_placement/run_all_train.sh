#!/bin/bash
# Run all TableQ4B placement ablation training experiments
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement"

echo "[$(date)] === Starting TableQ4B all train ==="

for PLACEMENT in B_qvlm C_attn D_mlp E_all; do
    echo "[$(date)] --- Placement $PLACEMENT ---"
    bash "$BASE/$PLACEMENT/train.sh"
done

echo "[$(date)] === All done ==="
