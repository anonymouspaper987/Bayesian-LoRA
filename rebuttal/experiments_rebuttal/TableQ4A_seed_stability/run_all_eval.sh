#!/bin/bash
# Run all TableQ4A seed stability evaluation experiments
set -e

BASE="/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4A_seed_stability"

echo "[$(date)] === Starting TableQ4A all eval ==="

for SEED in 456 789 2024; do
    echo "[$(date)] --- Seed $SEED ---"
    bash "$BASE/ARC-Easy/seed_$SEED/eval.sh"
done

echo "[$(date)] === All done ==="
