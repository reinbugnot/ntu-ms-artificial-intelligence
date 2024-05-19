#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=test
#SBATCH --output=./logs/output/output_%j.out     # replace with your own output path
#SBATCH --error=./logs/error/error_%j.err        # replace with your own error path

export CUBLAS_WORKSPACE_CONFIG=:16:8

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate acv                  # replace with your env name

python inference_swinfir.py \
--input './data/FFHQ/test/LQ' \
--output 'results/train_SwinFIR-T_SRx4_CUSTOM' \
--model_path 'experiments/train_SwinFIR-T_SRx4_CUSTOM_archived_20240426_112529/models/net_g_10000.pth' \
--task 'SwinFIR-T-CUSTOM' \
--training_patch_size 64 \
--scale 4 \




