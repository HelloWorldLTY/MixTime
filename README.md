# Mixtime: Predicting Immune Biomarkers with MultiModal Mixture-of-Expert Pathology Foundation Models Empowers Precision Oncology

Mixtime is a multi-expert pathology foundation-model fusion pipeline for predicting multiplexed immunofluorescence (mIF) protein expression from hematoxylin and eosin (H&E) images. The model extends a pretrained GigaTIME U-Net++ backbone with cross-attention experts from UNI v2, CONCH v1.5, and STPath features, and produces spatial 17-channel Orion biomarker expression maps.

This repository contains two complementary workflows:

- `train/`: training and evaluation code for HEMIT, Orion, and mixed HEMIT+Orion supervision.
- `finalmodel_mixtime/`: deployable inference pipeline for whole-slide images (WSIs), including tissue patch extraction, expert embedding extraction, and biomarker prediction.

The README is written as an executable methods guide: paths are examples, but the file formats, command-line flags, model inputs, and outputs reflect the code in this repository.

## Highlights

- Predicts 17 Orion biomarker channels from H&E image patches.
- Supports mixed supervision from Orion 17-channel mIF and HEMIT 3-channel IF labels.
- Uses three external expert feature sources: UNI v2, CONCH v1.5, and STPath.
- Supports expert ablations, cross-expert attention, dynamic gating, and multi-scale FiLM.
- Provides end-to-end WSI inference from `.tiff` slides to ranked biomarker JSON.
- Caches intermediate WSI embeddings, enabling failed runs to resume without recomputing all encoders.

## Repository Layout

```text
mixtime/
|-- README.md
|-- train/
|   |-- run_train.sh
|   |-- train_mixed_orion.py
|   |-- train_fusion_orion.py
|   |-- train_fusion_v2.py
|   |-- train_cross_attention.py
|   |-- train_baseline.py
|   |-- models/
|   |   |-- multi_expert_v2.py
|   |-- losses/
|   |   |-- channel_focal.py
|-- finalmodel_mixtime/
|   |-- README_inference.md
|   |-- run_wsi_inference.sh
|   |-- submit_wsi_inference.sh
|   |-- install_conch.sh
|   |-- install_stpath_deps.sh
|   |-- fix_torch.sh
|   |-- models/
|   |   |-- multi_expert_v2.py
|   |-- scripts/
|   |   |-- predict_sample_level.py
|   |   |-- predict_slide_report.py
|   |   |-- run_conch.py
|   |   |-- run_stpath.py
|   |   |-- run_univ2.py
|   |-- wsi_test/
|       |-- segment_patch.py
|       |-- extract_embeddings.py
|       |-- inference.py
```

## Model Overview

Mixtime takes an H&E image patch and expert embeddings for the same patch:

```text
H&E patch [B, 3, H, W]
  |
  |-- GigaTIME U-Net++ image backbone
  |
  |-- UNI v2 embedding       [B, 1536]
  |-- CONCH v1.5 embedding   [B, 768]
  |-- STPath feature         [B, 512] or [B, 38984]
  |
  |-- MultiExpertFusionV2
  |
  `-- mIF prediction         [B, 17, H, W]
```

The fusion model is implemented in `train/models/multi_expert_v2.py` and mirrored under `finalmodel_mixtime/models/` for inference. It starts from a pretrained GigaTIME U-Net++ backbone and injects expert information through cross-attention blocks. STPath gene-expression features are projected before attention when the feature dimension is large.

Optional components can be enabled at training time:

- `--use_cross_expert`: self-attention among active experts before fusion.
- `--use_dynamic_gating`: input-dependent expert weighting instead of fixed learned weights.
- `--use_multiscale_film`: FiLM modulation in multiple U-Net++ decoder stages.
- `--disable_uni`, `--disable_conch`, `--disable_stpath`: expert ablations.

## Biomarker Channels

The model predicts channels in the following fixed order:

| Index | Biomarker | Index | Biomarker |
|---:|---|---:|---|
| 0 | Hoechst | 9 | PD-L1 |
| 1 | CD31 | 10 | CD3e |
| 2 | CD45 | 11 | CD163 |
| 3 | CD68 | 12 | E-cadherin |
| 4 | CD4 | 13 | PD-1 |
| 5 | FOXP3 | 14 | Ki67 |
| 6 | CD8a | 15 | Pan-CK |
| 7 | CD45RO | 16 | SMA |
| 8 | CD20 |  |  |

For mixed HEMIT+Orion training, HEMIT labels are mapped into this 17-channel space:

| HEMIT channel | Orion channel |
|---|---|
| panCK | Pan-CK |
| CD3 | CD3e |
| DAPI | Hoechst |

Only mapped HEMIT channels contribute to the masked loss.

## Environment

The code is written for Python/PyTorch GPU environments. Exact package versions may depend on the cluster image used for training and inference, but the following packages are required by the scripts:

```bash
conda create -n mixtime python=3.10 -y
conda activate mixtime

pip install torch torchvision torchaudio
pip install numpy pandas pillow scipy tqdm tifffile albumentations
pip install timm==0.9.16 huggingface_hub wandb
pip install anndata scanpy einops
pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```

Inference through `finalmodel_mixtime/run_wsi_inference.sh` also expects local copies of Trident and STPath, which are referenced in `PYTHONPATH` by the shell script.

Several encoders are gated on Hugging Face Hub. Export tokens at runtime; do not commit tokens to the repository.

```bash
export HF_TOKEN="<your_huggingface_token>"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="/path/to/hf_cache"
export HF_HUB_CACHE="/path/to/hf_cache"
```

Gated or external models used by the inference pipeline include:

| Component | Model/source | Role |
|---|---|---|
| GigaTIME | `prov-gigatime/GigaTIME` | pretrained U-Net++ backbone |
| GigaPath | `prov-gigapath/prov-gigapath` | patch encoder for STPath image features |
| UNI v2 | `MahmoodLab/UNI2-h` | expert embedding |
| CONCH v1.5 | `MahmoodLab/conchv1_5` | expert embedding |
| STPath | local STPath checkout and weights | spatial transcriptomic feature prediction |

## Data Requirements

### Orion Dataset

`train_mixed_orion.py` and `train_fusion_orion.py` expect the Orion directory to contain:

```text
ORION_ROOT/
|-- train_dataframe.csv
|-- val_dataframe.csv
|-- test_dataframe.csv
|-- he/
|-- if/
```

Each CSV must include image and target paths used by the dataset loader:

- `image_path`: relative path to an H&E image, usually under `he/`.
- `target_path`: relative path to a 17-channel mIF TIFF, usually under `if/`.

Optional preprocessed labels can be supplied with `--npy_dir`. When provided, the loader first searches for a `.npy` version of each Orion target and falls back to TIFF if the `.npy` file is missing.

### HEMIT Dataset

Mixed training expects HEMIT data in this structure:

```text
HEMIT_ROOT/
|-- train_data.csv
|-- val_data.csv
|-- test_data.csv
|-- he/
|   |-- train/
|   |   |-- input/
|   |   |-- label/
|   |-- val/
|   |   |-- input/
|   |   |-- label/
|   |-- test/
|       |-- input/
|       |-- label/
```

Each HEMIT CSV must contain a `name` column. The loader resolves input and label files as:

```text
HEMIT_ROOT/he/<split>/input/<name>.tif
HEMIT_ROOT/he/<split>/label/<name>.tif
```

### Expert Embeddings

Training scripts expect precomputed embeddings under `--emb_dir`:

```text
EMB_ROOT/
|-- univ2emb/
|   |-- orion_train.pkl
|   |-- orion_val.pkl
|   |-- orion_test.pkl
|   |-- meiphi_train.pkl
|   |-- meiphi_val.pkl
|   |-- meiphi_test.pkl
|-- conchemb/
|   |-- orion_train.pkl
|   |-- orion_val.pkl
|   |-- orion_test.pkl
|   |-- meiphi_train.pkl
|   |-- meiphi_val.pkl
|   |-- meiphi_test.pkl
|-- stpathinfo/
    |-- orion_train.pkl
    |-- orion_val.pkl
    |-- orion_test.pkl
    |-- meiphi_train.pkl
    |-- meiphi_val.pkl
    |-- meiphi_test.pkl
```

Expected feature dimensions:

| Feature | Dimension | Notes |
|---|---:|---|
| UNI v2 | 1536 | stored as PyTorch tensors |
| CONCH v1.5 | 768 | stored as PyTorch tensors |
| STPath embedding | 512 | selected with `--stpath_feature_type emb` |
| STPath gene prediction | about 30k, commonly 38984 | selected with `--stpath_feature_type gene` |

The STPath pickle must contain either an `emb` key or a `pred` key depending on `--stpath_feature_type`.

## Training

### Recommended Mixed HEMIT+Orion Training

The primary publication-oriented training script is `train/train_mixed_orion.py`. It trains a 17-channel model while allowing HEMIT samples to supervise only the three mapped channels.

```bash
cd train

python train_mixed_orion.py \
  --hemit_dir /path/to/HEMIT \
  --orion_dir /path/to/ORION/ORIONCRC_dataset_tile_20x \
  --emb_dir /path/to/embedding_root \
  --gigatime_weights /path/to/GigaTIME/model.pth \
  --npy_dir /path/to/orion_npy_labels \
  --freeze_gigatime true \
  --num_heads 4 \
  --num_tokens 8 \
  --dropout 0.1 \
  --stpath_feature_type gene \
  --loss_type masked_mix_pearson \
  --batch_size 16 \
  --epochs 200 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --patience 50 \
  --val_every 5 \
  --num_workers 0 \
  --output_dir /path/to/outputs \
  --exp_name orion_gene_mixpearson \
  --wandb_project gigatime-mixed \
  --seed 42
```

Use `--no_wandb` to disable Weights & Biases logging:

```bash
python train_mixed_orion.py ... --no_wandb
```

If `--gigatime_weights` is omitted, the script attempts to download `prov-gigatime/GigaTIME` from Hugging Face.

### SLURM Training

`train/run_train.sh` is a template SLURM launcher. Update the partition, paths, conda activation, and `PYTHONPATH` before submission:

```bash
cd train
sbatch run_train.sh
```

The included template runs `train_cross_attention.py`; for the mixed 17-channel model, replace the command with `train_mixed_orion.py` and the arguments shown above.

### Orion-Only Training

Use `train/train_fusion_orion.py` when training only on Orion 17-channel labels:

```bash
cd train

python train_fusion_orion.py \
  --orion_dir /path/to/ORION/ORIONCRC_dataset_tile_20x \
  --emb_dir /path/to/embedding_root \
  --gigatime_weights /path/to/GigaTIME/model.pth \
  --stpath_feature_type gene \
  --loss_type mix_pearson \
  --batch_size 16 \
  --epochs 200 \
  --lr 1e-4 \
  --output_dir /path/to/outputs \
  --exp_name orion_gene_mixpearson
```

### HEMIT-Only and Baseline Training

The repository also includes:

- `train/train_fusion_v2.py`: HEMIT-focused multi-expert fusion training.
- `train/train_cross_attention.py`: earlier cross-attention fusion training.
- `train/train_baseline.py`: image-only GigaTIME baseline without external embeddings.

These scripts are useful for ablations and comparisons.

### Loss Functions

Losses are implemented in `train/losses/channel_focal.py`.

| `--loss_type` | Description |
|---|---|
| `smooth_l1` | pixelwise Smooth L1 loss |
| `focal` | channel focal weighting based on per-channel error |
| `weighted` | fixed channel-weighted loss |
| `mix_pearson` | Smooth L1 plus channel and pixel Pearson correlation losses |
| `masked_mix_pearson` | mixed-dataset version of `mix_pearson` that respects channel masks |

For mixed HEMIT+Orion training, `masked_mix_pearson` is the natural default because it prevents unavailable HEMIT channels from contributing to the objective.

### Checkpoints and Results

Training writes outputs to:

```text
<output_dir>/<exp_name>/
|-- best_model.pth
|-- checkpoint_epoch_<N>.pth
|-- results.json
```

`best_model.pth` stores:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `val_loss`
- `val_pcc_mean` when available
- `config`, containing all training arguments

`results.json` stores the best epoch, validation loss, test losses, Orion PCC/SCC metrics, HEMIT PCC/SCC metrics when available, and the model configuration.

### Resume or Warm Start

Resume a full interrupted run, including optimizer and scheduler:

```bash
python train_mixed_orion.py \
  ... \
  --resume /path/to/outputs/orion_gene_mixpearson/checkpoint_epoch_50.pth
```

Warm start from a trained model while reinitializing optimizer and scheduler:

```bash
python train_mixed_orion.py \
  ... \
  --pretrained_model /path/to/outputs/orion_gene_mixpearson/best_model.pth
```

## Whole-Slide Inference

The deployable WSI pipeline lives in `finalmodel_mixtime/run_wsi_inference.sh`.

```bash
cd finalmodel_mixtime

bash run_wsi_inference.sh /path/to/slide.tiff /path/to/output_dir 0
```

Arguments:

```text
bash run_wsi_inference.sh <slide_path> [output_dir] [gpu_id]
```

- `<slide_path>`: input H&E WSI, typically `.tiff`.
- `[output_dir]`: output directory; defaults to `finalmodel_mixtime/wsi_test`.
- `[gpu_id]`: CUDA device index; defaults to `0`.

Before running, edit these paths in `run_wsi_inference.sh` for your system:

```bash
GIGATIME_WEIGHTS="/path/to/GigaTIME/model.pth"
CHECKPOINT="/path/to/mixtime/best_model.pth"
STPATH_ROOT="/path/to/STPath"
HF_CACHE="/path/to/hf_cache"
```

### SLURM WSI Inference

`finalmodel_mixtime/submit_wsi_inference.sh` is a SLURM wrapper around `run_wsi_inference.sh`.

Before using it, update:

- `#SBATCH --partition`
- `#SBATCH --gres`
- `#SBATCH --output`
- `#SBATCH --error`
- `SCRIPT_DIR`
- input slide path
- output directory
- Hugging Face token environment variables

Submit with:

```bash
cd finalmodel_mixtime
sbatch submit_wsi_inference.sh
```

Do not hard-code personal tokens in scripts intended for publication. Prefer environment variables, scheduler secrets, or a private shell profile.

## WSI Inference Pipeline Details

`run_wsi_inference.sh` creates three Python scripts inside the output directory and executes them in order.

### Step 1: Tissue Segmentation and Patch Extraction

Generated script:

```text
<output_dir>/segment_patch.py
```

The script uses Trident to:

1. load the WSI,
2. segment tissue with the selected segmenter, default `hest`,
3. extract patch coordinates at 20x magnification,
4. dump 256 x 256 PNG patches with no overlap.

Patch names encode level-0 coordinates:

```text
000001_x<X>_y<Y>.png
```

Coordinates are used downstream to preserve the spatial order required by STPath.

### Step 2: Expert Embedding Extraction

Generated script:

```text
<output_dir>/extract_embeddings.py
```

The script scans patches in sorted `(x, y)` order and writes:

```text
<output_dir>/embeddings/
|-- gigapath.pt
|-- univ2.pt
|-- conch.pt
|-- stpath_gene.npy
|-- coords.npy
|-- patch_paths.txt
```

Embedding outputs:

| File | Shape | Purpose |
|---|---:|---|
| `gigapath.pt` | `[N, 1536]` | image features for STPath |
| `univ2.pt` | `[N, 1536]` | UNI expert |
| `conch.pt` | `[N, 768]` | CONCH expert |
| `stpath_gene.npy` | `[N, 38984]` | STPath gene-expression expert |
| `coords.npy` | `[N, 2]` | patch coordinates in level-0 pixels |

Existing embedding files are reused, so the pipeline can resume after failures.

### Step 3: Mixtime mIF Prediction

Generated script:

```text
<output_dir>/inference.py
```

For each patch, the model receives:

```text
patch image    [1, 3, 512, 512]
UNI feature    [1, 1536]
CONCH feature  [1, 768]
STPath feature [1, 38984]
```

The output is:

```text
prediction     [1, 17, 512, 512]
```

The script reports mean expression per biomarker for each patch and averages patch-level means into a slide-level ranking.

## Inference Output

Whole-slide inference writes:

```text
<output_dir>/<slide_name>_predictions.json
```

Example schema:

```json
{
  "n_patches": 87,
  "slide_level_ranked": [
    {
      "rank": 1,
      "biomarker": "PD-1",
      "mean_expression": 102.23
    }
  ],
  "per_patch": [
    {
      "patch_x": 12288,
      "patch_y": 4096,
      "mean_expression": {
        "Hoechst": 42.1,
        "CD31": 0.02,
        "CD45": 0.31
      }
    }
  ]
}
```

`slide_level_ranked` is suitable for high-level biomarker prioritization. `per_patch` preserves spatial coordinates for downstream visualization or spatial analysis.

## Patch-Level or Report-Image Inference

For pre-extracted H&E images with precomputed embeddings, use:

```bash
cd finalmodel_mixtime

python scripts/predict_slide_report.py \
  --checkpoint /path/to/best_model.pth \
  --gigatime_weights /path/to/GigaTIME/model.pth \
  --slide_dir /path/to/slide_report_images \
  --emb_dir /path/to/report_embedding \
  --output /path/to/predictions.json \
  --device cuda
```

The script expects image names to match embedding files:

```text
<name>.png
<name>_univ2.pkl
<name>_conch.pkl
<name>_stpath.pkl
```
