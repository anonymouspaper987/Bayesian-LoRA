#!/bin/bash
# Run all TableQ4A seed stability training experiments
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4A_seed_stability"

echo "[$(date)] === Starting TableQ4A all train ==="

for DATASET in ARC-Easy; do
    for SEED in 123 456 789 2024; do
        echo "[$(date)] --- Dataset $DATASET Seed $SEED ---"
        bash "$BASE/$DATASET/seed_$SEED/train.sh"
    done
done

echo "[$(date)] === All done ==="
