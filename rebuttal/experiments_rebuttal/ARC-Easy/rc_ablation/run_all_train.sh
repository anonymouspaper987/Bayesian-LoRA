#!/bin/bash
# Run all rc ablation training experiments (r ≠ c)
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/ARC-Easy/rc_ablation"

echo "[$(date)] === Starting rc ablation train ==="

for RC in r4_c9 r9_c4 r16_c9 r9_c16; do
    echo "[$(date)] --- $RC ---"
    bash "$BASE/$RC/train.sh"
done

echo "[$(date)] === All done ==="
