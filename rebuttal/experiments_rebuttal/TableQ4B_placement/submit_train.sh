#!/bin/bash
#SBATCH --job-name=q4b-placement-train
#SBATCH --partition=ai
#SBATCH --account=cis250976-ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --exclude=h012,h014,h019,h020
#SBATCH --output=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/slurm_%j.out
#SBATCH --error=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/slurm_%j.err

set -x
bash /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/run_all_train.sh
