#!/bin/bash
#SBATCH --job-name=FDQN
#SBATCH --account=AICDI
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate SB3

python -u train_sb3.py > log.txt

