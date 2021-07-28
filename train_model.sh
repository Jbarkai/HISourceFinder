#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=vnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=9GB
 
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4
module load scikit-learn/0.22.2.post1-fosscuda-2019b-Python-3.7.4
 
source /data/$USER/.envs/vnet/bin/activate

python3 train_model.py --nEpochs=2 --terminal_show_freq=25 --scale=loud --k_folds=1 --subsample=1

deactivate