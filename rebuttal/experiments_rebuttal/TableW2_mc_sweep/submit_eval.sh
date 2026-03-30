#!/bin/bash
#SBATCH --job-name=w2-mc-sweep-eval
#SBATCH --partition=ai
#SBATCH --account=cis260245-ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --exclude=h012,h014,h019,h020
#SBATCH --output=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableW2_mc_sweep/slurm_%j.out
#SBATCH --error=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableW2_mc_sweep/slurm_%j.err

set -x
bash /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableW2_mc_sweep/eval_sweep.sh
