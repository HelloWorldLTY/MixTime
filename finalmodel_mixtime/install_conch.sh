#!/bin/bash
#SBATCH --job-name=install_conch
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:a40:1
#SBATCH --mem=8G --time=15:00
#SBATCH --output=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/install_conch_%j.out
#SBATCH --error=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/install_conch_%j.err

PIP=/gpfs/radev/home/tl688/.conda/envs/trident/bin/pip
echo "Installing CONCH..."
${PIP} install "git+https://github.com/Mahmoodlab/CONCH.git" 2>&1 | tail -5
echo "Done."
${PIP} show conch 2>/dev/null | head -3
