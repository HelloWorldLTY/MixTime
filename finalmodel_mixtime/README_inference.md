# GigaTIME WSI Inference Pipeline

End-to-end pipeline for inferring 17-channel multiplexed immunofluorescence (mIF) protein expression from a hematoxylin & eosin (H&E) whole-slide image (WSI), using the trained `MultiExpertFusionV2` model.

---

## Overview

The pipeline has three sequential steps:

```
WSI (.tiff)
  │
  ▼ Step 1 — Trident: tissue segmentation + patch extraction
  │           87 × 256-px PNG patches @ 20×  (coords in filename)
  │
  ▼ Step 2 — Embedding extraction (4 encoders)
  │           GigaPath  [N, 1536]  → used as STPath image input
  │           UNIv2     [N, 1536]  → expert 1
  │           CONCH v1.5 [N,  768] → expert 2
  │           STPath    [N,38984]  → gene-expression predictions (expert 3)
  │
  ▼ Step 3 — MultiExpertFusionV2 inference
              per-patch [17, 256, 256] protein expression maps
              → slide-level ranked biomarker table + JSON output
```

---

## Environment

### Conda environment

All steps run inside the `trident` conda environment:

```bash
conda activate trident
# or use the full python path:
/gpfs/radev/home/tl688/.conda/envs/trident/bin/python
```

### Key packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | — |
| PyTorch | 2.11.0+cu128 | Deep learning framework |
| CUDA (compiled) | 12.8 | GPU backend |
| timm | 0.9.16 | Required by GigaPath (`prov-gigapath`) |
| conch | 0.1.0 | CONCH v1.5 vision encoder |
| anndata | 0.11.4 | STPath dependency |
| scanpy | 1.11.5 | STPath dependency |
| trident | local | WSI segmentation + patching |

> **Note:** GigaPath requires exactly `timm==0.9.16`. Do not upgrade timm without testing GigaPath compatibility.

### One-time package installs (already done)

```bash
pip install "git+https://github.com/Mahmoodlab/CONCH.git"   # conch 0.1.0
pip install anndata scanpy einops                             # STPath deps
```

### HuggingFace authentication

Several models are gated on HuggingFace Hub. The token must be available at runtime:

```bash
export HF_TOKEN=<your_hf_token>
export HUGGING_FACE_HUB_TOKEN=<your_hf_token>
export HF_HOME=/gpfs/radev/home/tl688/scratch
export HF_HUB_CACHE=/gpfs/radev/home/tl688/scratch
```

Gated models used:

| Model | HF repo |
|-------|---------|
| GigaPath (patch encoder) | `prov-gigapath/prov-gigapath` |
| UNI v2 | `MahmoodLab/UNI2-h` |
| CONCH v1.5 | `MahmoodLab/conchv1_5` |

---

## Repository Layout

```
finalmodel/
├── run_wsi_inference.sh          # Main pipeline entry point
├── submit_wsi_inference.sh       # SLURM job submission wrapper
├── train_mixed_orion.py          # Training script (reference)
├── models/
│   └── multi_expert_v2.py        # MultiExpertFusionV2 architecture
├── orion_gene_mixpearson/
│   └── best_model.pth            # Trained model checkpoint (epoch 8)
├── e48822b5419308cf918ae920239408d7b33327fa/
│   └── model.pth                 # Pretrained GigaTIME U-Net++ backbone
├── trident/                      # Trident WSI toolkit (tissue seg + patching)
└── wsi_test/                     # Default output directory
    ├── segment_patch.py          # Step 1 script (auto-generated)
    ├── extract_embeddings.py     # Step 2 script (auto-generated)
    ├── inference.py              # Step 3 script (auto-generated)
    ├── patches/                  # Extracted patch PNGs
    ├── embeddings/               # Per-encoder embedding files
    │   ├── gigapath.pt           # [N, 1536] float32
    │   ├── univ2.pt              # [N, 1536] float32
    │   ├── conch.pt              # [N,  768] float32
    │   ├── stpath_gene.npy       # [N,38984] float32
    │   └── coords.npy            # [N, 2]   int64 (level-0 pixel x, y)
    └── <slide_name>_predictions.json
```

### External dependency

STPath model and vocabulary are at:

```
/gpfs/radev/project/zhao/tl688/STPath/
├── STPath/stfm.pth               # STPath model weights
└── utils_data/symbol2ensembl.json  # Gene vocabulary
```

---

## Usage

### Basic run (submits SLURM job automatically)

```bash
cd /gpfs/radev/project/ying_rex/tl688/GigaTIME/finalmodel
bash run_wsi_inference.sh <path/to/slide.tiff> [output_dir] [gpu_id]
```

Example:

```bash
bash run_wsi_inference.sh \
    /gpfs/radev/home/tl688/pitl688/reg2026/reg2026/train/PIT_06_01555_01.tiff
```

Output goes to `./wsi_test/` by default.

### Submit to SLURM (recommended for large slides)

```bash
sbatch submit_wsi_inference.sh
```

The submit script targets the `gpu_devel` partition (A40 GPU, 16 GB RAM). Edit `submit_wsi_inference.sh` to change the partition or resources.

---

## Step-by-step Details

### Step 1 — Tissue segmentation and patching (`segment_patch.py`)

Uses the [Trident](https://github.com/mahmoodlab/trident) toolkit:

1. **Tissue segmentation** with the HEST segmentation model at its native resolution.
2. **Patch coordinate extraction** at 20× magnification, 256 × 256 px, no overlap.
3. **Dump patches as PNG** to `<output_dir>/patches/<slide_name>/`.

Patch filename format:
```
{index:06d}_x{X}_y{Y}.png
```
where `X`, `Y` are the **level-0 pixel coordinates** of the top-left corner. These coordinates are parsed by downstream steps to recover spatial positions.

### Step 2 — Embedding extraction (`extract_embeddings.py`)

Processes all patches in order sorted by `(x, y)`. Each encoder result is saved to disk; if the file already exists, it is loaded from cache (safe to resume after a crash).

| Step | Encoder | Input | Output | Notes |
|------|---------|-------|--------|-------|
| 1/4 | **GigaPath** | 256 px PNG | `[N, 1536]` | Required by STPath |
| 2/4 | **UNIv2** | 256 px PNG | `[N, 1536]` | Expert 1 |
| 3/4 | **CONCH v1.5** | 256 px PNG | `[N, 768]` | Expert 2; must use v1.5, not v1 |
| 4/4 | **STPath** | coords + GigaPath feats | `[N, 38984]` | Predicts gene expression |

Half-precision encoders (GigaPath, UNIv2, CONCH v1.5) use `torch.autocast` to avoid dtype mismatches with `float32` model biases.

**STPath** takes all patches collectively (not one at a time):
```python
pred, xout = agent.inference(
    coords=coords,           # [N, 2] level-0 pixel coords
    img_features=feats_gp    # [N, 1536] GigaPath features
)
# pred: [N, 38984] gene-expression predictions
```
Spatial coordinates are normalised internally by STPath.

### Step 3 — mIF inference (`inference.py`)

Loads `orion_gene_mixpearson/best_model.pth` (MultiExpertFusionV2, epoch 8).

For each patch:
```
patch image  [1, 3, 512, 512]  (resized from 256 px, ImageNet-normalised)
emb_uni      [1, 1536]
emb_conch    [1,  768]
emb_stpath   [1,38984]
     ↓
MultiExpertFusionV2
     ↓
pred         [1, 17, 512, 512]  (Orion 17 biomarker channels, Softplus output)
```

Per-patch mean expression is averaged spatially across all `512×512` pixels to give a scalar per biomarker per patch. Slide-level expression is the mean across all patches.

---

## Output Format

`<slide_name>_predictions.json`:

```json
{
  "n_patches": 87,
  "slide_level_ranked": [
    {"rank": 1, "biomarker": "PD-1",    "mean_expression": 102.23},
    {"rank": 2, "biomarker": "Hoechst", "mean_expression": 35.92},
    ...
  ],
  "per_patch": [
    {
      "patch_x": 12288,
      "patch_y": 4096,
      "mean_expression": {
        "Hoechst": 42.1, "CD31": 0.02, "CD45": 0.31, ...
      }
    },
    ...
  ]
}
```

### 17 Orion biomarker channels (in model order)

| Index | Biomarker | Index | Biomarker |
|-------|-----------|-------|-----------|
| 0 | Hoechst | 9 | PD-L1 |
| 1 | CD31 | 10 | CD3e |
| 2 | CD45 | 11 | CD163 |
| 3 | CD68 | 12 | E-cadherin |
| 4 | CD4 | 13 | PD-1 |
| 5 | FOXP3 | 14 | Ki67 |
| 6 | CD8a | 15 | Pan-CK |
| 7 | CD45RO | 16 | SMA |
| 8 | CD20 | | |

---

## Model Checkpoint Details

| | |
|---|---|
| File | `orion_gene_mixpearson/best_model.pth` |
| Epoch | 8 |
| Val loss | 2.405 |
| Test PCC (mean) | 0.246 |
| Best channel PCC | Hoechst 0.870, PD-1 0.486 |
| Architecture | MultiExpertFusionV2 |
| STPath feature type | gene (38984-dim) |
| Loss | mix_pearson |
| Experts | UNI=1536, CONCH=768, STPath=38984 |
| GigaTIME backbone | frozen during training: `False` |

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: NVIDIA driver too old (found version 12080)` | torch compiled for CUDA 13.0 but driver ≤ 570 supports only CUDA 12.8 | Run on GPU node (not login node); torch 2.11.0+cu128 is already installed |
| `Input type (Half) and bias type (float) should be the same` | Half-precision encoder on float32 model | Resolved via `torch.autocast` in `batch_encode()` |
| `Please install CONCH` | `conch` package missing | `pip install git+https://github.com/Mahmoodlab/CONCH.git` |
| `ModuleNotFoundError: anndata` | STPath dependency missing | `pip install anndata scanpy einops` |
| `mat1 and mat2 shapes cannot be multiplied (1x512 and 768x512)` | Using `conch_v1` (512-dim) instead of `conch_v15` (768-dim) | Script already uses `conch_v15` |
| `Cannot access gated repo` | HF token not set | Export `HF_TOKEN` before running |

---

## Example Result — PIT_06_01555_01.tiff

Slide: pituitary adenoma (20× H&E), 87 tissue patches extracted.

| Rank | Biomarker | Mean Expression |
|------|-----------|----------------|
| 1 | PD-1 | 102.23 |
| 2 | Hoechst | 35.92 |
| 3 | SMA | 1.22 |
| 4 | Pan-CK | 0.45 |
| 5 | CD45 | 0.32 |
| 6 | E-cadherin | 0.20 |
| 7 | PD-L1 | 0.04 |
| 8–17 | CD163, CD68, … | < 0.03 |
