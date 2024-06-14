#!/bin/bash 
#SBATCH --job-name=gpu_TARS
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-8-v100
#SBATCH --nodelist=gpunode0
#SBATCH --gres=gpu:1
#_SBATCH --exclusive

export WANDB_API_KEY=eb9398a48f1f6ce3c1b82963139e0202df39ce2a

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

srun python test.py
