#!/bin/bash

#SBATCH -J Audio_Diffusion              # Job Name
#SBATCH -p batch			# (debug or batch)
#SBATCH --exclusive
#SBATCH -o runjob.out			# Output file name
#SBATCH --gres=gpu:8			# Number of GPUs to use

#source ~/MyEnv/bin/activate		# Activate my Python Env
#python -m eval.compute_model_stats
torchrun --nproc_per_node=8 --master-port 29505 src/train.py
