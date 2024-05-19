#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=(O_O)/
#SBATCH --output=./logs/output/output_%j.out     # replace with your own output path
#SBATCH --error=./logs/error/error_%j.err        # replace with your own error path

export CUBLAS_WORKSPACE_CONFIG=:16:8

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate acv-swinfir                  
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR-T_SRx4_CUSTOM.yml --auto_resume

# CUDA_VISIBLE_DEVICES=0 \
# replace with your env name
#python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx2_from_scratch.yml --launcher pytorch --auto_resume   #provide relative paths for both the 'train.py' and 'train...yml' files
