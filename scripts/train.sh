#!/bin/sh

#SBATCH --job-name=train_stct
#SBATCH --output=.logs/train_stct_%j.out
#SBATCH --output=.logs/train_stct_%j.err
#SBATCH --nodes=1

srun --gres=gpu:1 -p gpi.develop --time 01:59:00 --mem 10G python ../src/model.py