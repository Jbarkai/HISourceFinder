#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=vnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=9GB
 
module load Python/3.8.6-GCCcore-10.2.0
module load cuDNN/5.0-CUDA-7.5.18
 
source /data/$USER/.envs/vnet/bin/activate
 
python3 python train_model.py --nEpochs=2 --terminal_show_freq=25 --scale=loud --k_folds=1 --subsample=1

deactivate