#!/bin/bash
#SBATCH --job-name=fix_torch
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/fix_torch_%j.out
#SBATCH --error=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/fix_torch_%j.err

PIP=/gpfs/radev/home/tl688/.conda/envs/trident/bin/pip
PYTHON=/gpfs/radev/home/tl688/.conda/envs/trident/bin/python

echo "Current torch:"
${PYTHON} -c "import torch; print(torch.__version__)"

echo "Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Try cu128 first (matches driver 570 = CUDA 12.8)
echo "Trying torch cu128..."
${PIP} install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall 2>&1 | tail -5

echo "After install:"
${PYTHON} -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
