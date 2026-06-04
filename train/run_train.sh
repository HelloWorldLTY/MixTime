#!/bin/bash
#SBATCH --job-name=cross_attn_train
#SBATCH --partition=xxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust to your environment name)
# source ~/.bashrc
# conda activate gigatime

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/work/nvme/bdxk/zwang92/GigaTIME"

# Navigate to project directory
cd /work/nvme/bdxk/zwang92/GigaTIME

# Run training
python train_cross_attention.py \
    --hemit_dir /work/nvme/bdxk/hzhao11/HEMIT \
    --emb_dir /work/nvme/bdxk/zwang92/GigaTIME \
    --model_type cross_attention \
    --freeze_gigatime \
    --num_heads 4 \
    --num_tokens 16 \
    --batch_size 16 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 50 \
    --num_workers 4 \
    --output_dir /work/nvme/bdxk/zwang92/GigaTIME/outputs \
    --exp_name cross_attn_fusion \
    --wandb_project gigatime-fusion \
    --seed 42

echo "Training completed!"
