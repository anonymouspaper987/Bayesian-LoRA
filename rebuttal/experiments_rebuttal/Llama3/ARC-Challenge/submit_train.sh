#!/bin/bash
#SBATCH --job-name=llama3-arc-c-train
#SBATCH --partition=ai
#SBATCH --account=cis260245-ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --exclude=h012,h014,h019,h020
#SBATCH --output=/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Challenge/slurm_train_%j.out
#SBATCH --error=/home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Challenge/slurm_train_%j.err

set -x
bash /home/x-qlan1/rebuttal/Bayesian_lora_peft/rebuttal/Llama3/ARC-Challenge/run_train.sh
