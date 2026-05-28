#!/bin/bash
#SBATCH --job-name=gigatime_wsi
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/slurm_%j.out
#SBATCH --error=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel/wsi_test/slurm_%j.err

SCRIPT_DIR=/gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/trident:/gpfs/radev/project/zhao/tl688/STPath:${PYTHONPATH:-}"
export HF_HOME=/gpfs/radev/home/tl688/scratch
export HF_HUB_CACHE=/gpfs/radev/home/tl688/scratch
export HF_TOKEN=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
export HUGGING_FACE_HUB_TOKEN=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

cd "${SCRIPT_DIR}"
bash run_wsi_inference.sh \
    /gpfs/radev/home/tl688/pitl688/reg2026/reg2026/train/PIT_06_01555_01.tiff \
    "${SCRIPT_DIR}/wsi_test" \
    0
