#!/bin/bash
#SBATCH --job-name=DuelDDQN
#SBATCH --account=AICDI
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate SB3

rm slurm-*
python -u api_test.py > Double_DQN/log.txt

