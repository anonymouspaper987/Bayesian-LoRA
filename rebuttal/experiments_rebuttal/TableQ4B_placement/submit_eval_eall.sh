#!/bin/bash
#SBATCH --job-name=q4b-eall-eval
#SBATCH --partition=ai
#SBATCH --account=cis260245-ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --exclude=h012,h014,h019,h020
#SBATCH --output=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/slurm_eall_%j.out
#SBATCH --error=/home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/slurm_eall_%j.err

set -x
bash /home/anonymous/rebuttal/Bayesian_lora_peft/rebuttal/TableQ4B_placement/E_all/eval.sh
