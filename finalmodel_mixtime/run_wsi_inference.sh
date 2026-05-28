#!/bin/bash
# -*- coding: utf-8 -*-
# =============================================================================
# GigaTIME WSI mIF Inference Pipeline
#
# Usage:  bash run_wsi_inference.sh <path/to/slide.tiff>
#
# Steps:
#   1. Segment tissue + extract patches  (trident)
#   2. Extract UNIv2 / GigaPath / CONCH / STPath embeddings
#   3. Per-patch mIF inference with MultiExpertFusionV2
# =============================================================================

set -euo pipefail

# --------------- paths --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GIGATIME_WEIGHTS="${SCRIPT_DIR}/e48822b5419308cf918ae920239408d7b33327fa/model.pth"
CHECKPOINT="${SCRIPT_DIR}/orion_gene_mixpearson/best_model.pth"
STPATH_ROOT="/gpfs/radev/project/zhao/tl688/STPath"
HF_CACHE="/gpfs/radev/home/tl688/scratch"

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/trident:${STPATH_ROOT}:${PYTHONPATH:-}"

# --------------- arguments ----------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <slide_path> [output_dir] [gpu_id]"
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/wsi_test}"
GPU="${3:-0}"

SLIDE_STEM="$(basename "${INPUT}" | sed 's/\.[^.]*$//')"
PATCH_DIR="${OUTPUT_DIR}/patches"
EMB_DIR="${OUTPUT_DIR}/embeddings"
RESULT_FILE="${OUTPUT_DIR}/${SLIDE_STEM}_predictions.json"

mkdir -p "${PATCH_DIR}" "${EMB_DIR}"

echo "============================================================"
echo " GigaTIME WSI Inference"
echo " Slide : ${INPUT}"
echo " Output: ${OUTPUT_DIR}"
echo " GPU   : ${GPU}"
echo "============================================================"

# =============================================================================
# Write Python scripts into the output directory so they can be inspected
# =============================================================================

# --------------------------------------------------------------------------- #
# segment_patch.py                                                             #
# --------------------------------------------------------------------------- #
cat > "${OUTPUT_DIR}/segment_patch.py" << 'PYEOF'
"""
Step 1: Tissue segmentation and patch extraction using Trident.

Saves patches as PNG files named  {slide_name}/{idx:06d}_x{X}_y{Y}.png
where X, Y are the level-0 pixel coordinates of each patch's top-left corner.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,  help="Path to WSI file")
    parser.add_argument("--patch_dir",  required=True,  help="Root dir to save patch PNGs")
    parser.add_argument("--job_dir",    required=True,  help="Trident working dir (for contours etc.)")
    parser.add_argument("--mag",        type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--segmenter",  default="hest", choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()

    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory

    os.makedirs(args.job_dir,   exist_ok=True)
    os.makedirs(args.patch_dir, exist_ok=True)

    with load_wsi(slide_path=args.input, lazy_init=False) as slide:
        # --- tissue segmentation ---
        seg_device = "cpu" if args.segmenter == "otsu" else f"cuda:{args.gpu}"
        seg_model = segmentation_model_factory(
            args.segmenter, confidence_thresh=0.5
        )
        slide.segment_tissue(
            segmentation_model=seg_model,
            target_mag=seg_model.target_mag,
            job_dir=args.job_dir,
            device=seg_device,
        )
        print("Tissue segmentation done.")

        # --- patch coordinate extraction ---
        mag_str     = f"{float(args.mag):g}"
        save_coords = os.path.join(
            args.job_dir, f"{mag_str}x_{args.patch_size}px_0px_overlap"
        )
        coords_path = slide.extract_tissue_coords(
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords,
        )
        print(f"Patch coordinates saved to: {coords_path}")

        # --- dump PNGs (filename contains level-0 x, y coordinates) ---
        out_dir = slide.dump_patches(
            coords_path=coords_path,
            save_patches_dir=args.patch_dir,
            image_format="png",
        )
        print(f"Patch images saved to: {out_dir}")

        # persist coords path for downstream scripts
        marker = os.path.join(args.patch_dir, "coords_path.txt")
        with open(marker, "w") as f:
            f.write(coords_path)

if __name__ == "__main__":
    main()
PYEOF

# --------------------------------------------------------------------------- #
# extract_embeddings.py                                                        #
# --------------------------------------------------------------------------- #
cat > "${OUTPUT_DIR}/extract_embeddings.py" << 'PYEOF'
"""
Step 2: Extract patch-level embeddings.

  • GigaPath  [N, 1536]  – used as image features for STPath
  • UNIv2     [N, 1536]  – used as expert embedding in GigaTIME
  • CONCH     [N,  768]  – used as expert embedding in GigaTIME
  • STPath    [N,38984]  – gene-expression predictions (via STPathInference)

Saves:
  <emb_dir>/gigapath.pt
  <emb_dir>/univ2.pt
  <emb_dir>/conch.pt
  <emb_dir>/stpath_gene.npy
  <emb_dir>/coords.npy          (patch (x, y) in level-0 pixels)
  <emb_dir>/patch_paths.txt
"""

import argparse
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---- helpers -----------------------------------------------------------------

def find_patches(patch_dir):
    """Return sorted list of (abs_path, x, y) for every PNG under patch_dir."""
    entries = []
    for root, _, files in os.walk(patch_dir):
        for fname in sorted(files):
            if not fname.endswith(".png"):
                continue
            m = re.search(r"_x(\d+)_y(\d+)\.png$", fname)
            if m:
                entries.append((
                    os.path.join(root, fname),
                    int(m.group(1)),
                    int(m.group(2)),
                ))
    entries.sort(key=lambda t: (t[1], t[2]))   # sort by (x, y)
    return entries


def batch_encode(encoder, patch_list, device, batch_size):
    """Run encoder over all patches; returns float32 CPU tensor [N, D]."""
    encoder.eval().to(device)
    transform  = encoder.eval_transforms
    precision  = encoder.precision
    all_feats  = []

    for start in tqdm(range(0, len(patch_list), batch_size), desc=f"  {encoder.enc_name}"):
        chunk = patch_list[start : start + batch_size]
        imgs  = []
        for path, _, _ in chunk:
            img = Image.open(path).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            if precision in (torch.float16, torch.bfloat16) and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=precision):
                    feats = encoder(batch.float()).float()
            else:
                feats = encoder(batch.to(precision)).float()
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


# ---- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir",  required=True)
    parser.add_argument("--emb_dir",    required=True)
    parser.add_argument("--stpath_root",required=True, help="Path to STPath repo root")
    parser.add_argument("--gpu",              type=int, default=0)
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--stpath_batch_size",type=int, default=256,
                        help="Number of patches per STPath inference chunk (256 recommended)")
    args = parser.parse_args()

    os.makedirs(args.emb_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # STPath repo must be on sys.path for imports
    if args.stpath_root not in sys.path:
        sys.path.insert(0, args.stpath_root)

    # ---- discover patches ----------------------------------------------------
    patches = find_patches(args.patch_dir)
    if len(patches) == 0:
        raise RuntimeError(f"No PNG patches found under {args.patch_dir}")
    print(f"Found {len(patches)} tissue patches.")

    coords = np.array([[x, y] for _, x, y in patches], dtype=np.int64)
    np.save(os.path.join(args.emb_dir, "coords.npy"), coords)

    paths_only = [p for p, _, _ in patches]
    with open(os.path.join(args.emb_dir, "patch_paths.txt"), "w") as f:
        f.write("\n".join(paths_only))

    # ---- import trident encoders ---------------------------------------------
    from trident.patch_encoder_models import encoder_factory

    # ===========================================================
    # GigaPath  (needed by STPath as img_features)
    # ===========================================================
    gp_path = os.path.join(args.emb_dir, "gigapath.pt")
    if os.path.exists(gp_path):
        print("\n[1/4] GigaPath  (cached)")
        feats_gp = torch.load(gp_path, map_location="cpu")
    else:
        print("\n[1/4] GigaPath")
        enc_gp   = encoder_factory("gigapath")
        feats_gp = batch_encode(enc_gp, patches, device, args.batch_size)
        torch.save(feats_gp, gp_path)
        print(f"      saved  gigapath.pt  {tuple(feats_gp.shape)}")
        del enc_gp;  torch.cuda.empty_cache()

    # ===========================================================
    # UNIv2
    # ===========================================================
    uni_path = os.path.join(args.emb_dir, "univ2.pt")
    if os.path.exists(uni_path):
        print("\n[2/4] UNIv2  (cached)")
        feats_uni = torch.load(uni_path, map_location="cpu")
    else:
        print("\n[2/4] UNIv2")
        enc_uni   = encoder_factory("uni_v2")
        feats_uni = batch_encode(enc_uni, patches, device, args.batch_size)
        torch.save(feats_uni, uni_path)
        print(f"      saved  univ2.pt     {tuple(feats_uni.shape)}")
        del enc_uni; torch.cuda.empty_cache()

    # ===========================================================
    # CONCH  (conch_v1, 768-dim)
    # ===========================================================
    conch_path = os.path.join(args.emb_dir, "conch.pt")
    if os.path.exists(conch_path):
        print("\n[3/4] CONCH  (cached)")
        feats_conch = torch.load(conch_path, map_location="cpu")
    else:
        print("\n[3/4] CONCH")
        enc_conch   = encoder_factory("conch_v15")
        feats_conch = batch_encode(enc_conch, patches, device, args.batch_size)
        torch.save(feats_conch, conch_path)
        print(f"      saved  conch.pt     {tuple(feats_conch.shape)}")
        del enc_conch; torch.cuda.empty_cache()

    # ===========================================================
    # STPath  – gene-expression prediction
    #   Input : coords [N, 2]  (level-0 pixel coords, normalised inside STPath)
    #           img_features [N, 1536]  (GigaPath features)
    #   Output: pred [N, 38984]  gene expression predictions
    # ===========================================================
    stp_path = os.path.join(args.emb_dir, "stpath_gene.npy")
    if os.path.exists(stp_path):
        print("\n[4/4] STPath  (cached)")
    else:
        print("\n[4/4] STPath gene-expression prediction")
        from stpath.app.pipeline.inference import STPathInference
        from stpath.data.dataset import rescale_coords

        gene_voc  = os.path.join(args.stpath_root, "utils_data", "symbol2ensembl.json")
        stfm_wts  = os.path.join(args.stpath_root, "STPath", "stfm.pth")

        agent = STPathInference(
            gene_voc_path=gene_voc,
            model_weight_path=stfm_wts,
            device=args.gpu,
        )

        # global coord normalisation
        coords_t = torch.from_numpy(coords.astype(np.float32)).to(args.gpu)
        coords_t[:, 0] = coords_t[:, 0] - coords_t[:, 0].min()
        coords_t[:, 1] = coords_t[:, 1] - coords_t[:, 1].min()
        coords_norm = rescale_coords(coords_t)

        feats_gp_np = feats_gp.numpy().astype(np.float32)
        N_patches   = len(coords_norm)
        bs          = args.stpath_batch_size

        organ_id = agent.tokenizer.organ_tokenizer.encode("Others", align_first=True)

        all_preds = []
        for start in tqdm(range(0, N_patches, bs), desc="  STPath"):
            end    = min(start + bs, N_patches)
            actual = end - start

            c_chunk   = coords_norm[start:end]
            f_chunk   = torch.from_numpy(feats_gp_np[start:end]).to(args.gpu)
            masked_ge = agent._generate_masked_ge_tokens(actual).to(args.gpu)
            tech_ids  = agent._generate_pad_tech_tokens(actual).to(args.gpu)
            organ_ids = torch.full((actual,), organ_id, dtype=torch.long, device=args.gpu)
            batch_idx = torch.zeros(actual, dtype=torch.long, device=args.gpu)

            # STPath model requires batch size == bs; pad last chunk if needed
            if actual < bs:
                pad       = bs - actual
                c_chunk   = torch.cat([c_chunk,   c_chunk[:1].expand(pad, -1)], dim=0)
                f_chunk   = torch.cat([f_chunk,   f_chunk[:1].expand(pad, -1)], dim=0)
                masked_ge = torch.cat([masked_ge, masked_ge[:1].expand(pad, -1)], dim=0)
                tech_ids  = torch.cat([tech_ids,  tech_ids[:1].expand(pad, -1)], dim=0)
                organ_ids = torch.cat([organ_ids, organ_ids[:1].expand(pad)], dim=0)
                batch_idx = torch.cat([batch_idx, batch_idx[:1].expand(pad)], dim=0)

            with torch.no_grad():
                pred_chunk, _ = agent.model.prediction_head(
                    img_tokens=f_chunk,
                    coords=c_chunk,
                    ge_tokens=masked_ge,
                    batch_idx=batch_idx,
                    tech_tokens=tech_ids,
                    organ_tokens=organ_ids,
                    return_all=True,
                )
            all_preds.append(pred_chunk[:actual].cpu())

        pred_np = torch.cat(all_preds, dim=0).numpy()
        np.save(stp_path, pred_np)
        print(f"      saved  stpath_gene.npy  {pred_np.shape}")


if __name__ == "__main__":
    main()
PYEOF

# --------------------------------------------------------------------------- #
# inference.py                                                                 #
# --------------------------------------------------------------------------- #
cat > "${OUTPUT_DIR}/inference.py" << 'PYEOF'
"""
Step 3: Per-patch mIF inference using MultiExpertFusionV2.

For every tissue patch the model predicts a [17, H, W] protein-expression map.
Outputs:
  • per-patch mean expression for each of the 17 Orion biomarkers
  • slide-level summary ranked by mean expression
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Orion 17 biomarker names (matches training order)
ORION_CHANNEL_NAMES = [
    "Hoechst", "CD31",   "CD45",        "CD68",      "CD4",   "FOXP3",
    "CD8a",    "CD45RO", "CD20",        "PD-L1",     "CD3e",  "CD163",
    "E-cadherin","PD-1", "Ki67",        "Pan-CK",    "SMA",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---- helpers -----------------------------------------------------------------

def find_patches(patch_dir):
    entries = []
    for root, _, files in os.walk(patch_dir):
        for fname in sorted(files):
            if not fname.endswith(".png"):
                continue
            m = re.search(r"_x(\d+)_y(\d+)\.png$", fname)
            if m:
                entries.append((
                    os.path.join(root, fname),
                    int(m.group(1)), int(m.group(2)),
                ))
    return sorted(entries, key=lambda t: (t[1], t[2]))


def preprocess(img_path, size=512):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)   # [1,3,H,W]


def load_model(ckpt_path, gigatime_weights, device):
    from models.multi_expert_v2 import create_multi_expert_v2

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    uni_dim   = 0 if cfg.get("disable_uni",    False) else 1536
    conch_dim = 0 if cfg.get("disable_conch",  False) else 768

    if cfg.get("disable_stpath", False):
        stpath_dim = 0
    else:
        # infer from saved weight shape  (emb_to_kv: [feat_dim*num_tokens*2, emb_dim])
        stpath_key = next(
            (k for k in ckpt["model_state_dict"]
             if "stpath" in k and "emb_to_kv.weight" in k),
            None
        )
        stpath_dim = (
            ckpt["model_state_dict"][stpath_key].shape[1]
            if stpath_key else 38984
        )

    model = create_multi_expert_v2(
        weights_path        = gigatime_weights,
        freeze_gigatime     = cfg.get("freeze_gigatime",   True),
        num_heads           = cfg.get("num_heads",            4),
        num_tokens          = cfg.get("num_tokens",           8),
        dropout             = cfg.get("dropout",            0.1),
        use_cross_expert    = cfg.get("use_cross_expert",  False),
        use_dynamic_gating  = cfg.get("use_dynamic_gating",False),
        use_multiscale_film = cfg.get("use_multiscale_film",False),
        uni_dim=uni_dim, conch_dim=conch_dim, stpath_dim=stpath_dim,
        out_channels=17,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model loaded  (epoch {ckpt.get('epoch','?')})")
    print(f"  UNI={uni_dim}  CONCH={conch_dim}  STPath={stpath_dim}")
    return model


# ---- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir",       required=True)
    parser.add_argument("--emb_dir",         required=True)
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--gigatime_weights",required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--gpu",             type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load model ----------------------------------------------------------
    model = load_model(args.checkpoint, args.gigatime_weights, device)

    # ---- load embeddings -----------------------------------------------------
    feats_uni   = torch.load(os.path.join(args.emb_dir, "univ2.pt"),     map_location="cpu")
    feats_conch = torch.load(os.path.join(args.emb_dir, "conch.pt"),     map_location="cpu")
    stpath_gene = torch.from_numpy(
        np.load(os.path.join(args.emb_dir, "stpath_gene.npy"))
    )

    # ---- discover patches (same order as extract_embeddings) ----------------
    patches = find_patches(args.patch_dir)
    N = len(patches)
    if N == 0:
        raise RuntimeError(f"No patches found under {args.patch_dir}")
    if N != feats_uni.shape[0]:
        raise RuntimeError(
            f"Patch count mismatch: {N} PNGs vs "
            f"{feats_uni.shape[0]} embeddings."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ---- per-patch inference -------------------------------------------------
    print(f"\nRunning inference on {N} patches ...")
    per_patch_results = []
    slide_sum = np.zeros(17, dtype=np.float64)

    for i, (path, x, y) in enumerate(tqdm(patches)):
        img   = preprocess(path).to(device)
        e_uni = feats_uni[i : i+1].to(device)
        e_con = feats_conch[i : i+1].to(device)
        e_stp = stpath_gene[i : i+1].to(device)

        with torch.no_grad():
            pred, _ = model(img, e_uni, e_con, e_stp)   # [1,17,H,W]

        mean_expr = pred[0].mean(dim=(1, 2)).cpu().numpy()   # [17]
        slide_sum += mean_expr

        per_patch_results.append({
            "patch_x": int(x),
            "patch_y": int(y),
            "mean_expression": {
                ORION_CHANNEL_NAMES[j]: float(mean_expr[j]) for j in range(17)
            },
        })

    # ---- slide-level summary -------------------------------------------------
    slide_mean = slide_sum / N
    ranked = sorted(
        enumerate(slide_mean), key=lambda t: t[1], reverse=True
    )

    print("\nSlide-level protein expression (top 17):")
    print(f"  {'Rank':<5} {'Biomarker':<15} {'Mean Expression':>16}")
    print("  " + "-" * 40)
    ranked_list = []
    for rank, (ch_idx, val) in enumerate(ranked, 1):
        name = ORION_CHANNEL_NAMES[ch_idx]
        print(f"  {rank:<5} {name:<15} {val:>16.4f}")
        ranked_list.append({"rank": rank, "biomarker": name,
                             "mean_expression": float(val)})

    output_data = {
        "n_patches": N,
        "slide_level_ranked": ranked_list,
        "per_patch": per_patch_results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
PYEOF

# =============================================================================
# Execute the pipeline
# =============================================================================

echo ""
echo "------------------------------------------------------------"
echo "Step 1: Segment tissue + extract patch PNGs"
echo "------------------------------------------------------------"
python "${OUTPUT_DIR}/segment_patch.py" \
    --input      "${INPUT}" \
    --patch_dir  "${PATCH_DIR}" \
    --job_dir    "${OUTPUT_DIR}/trident_work" \
    --mag        20 \
    --patch_size 256 \
    --segmenter  hest \
    --gpu        "${GPU}"

echo ""
echo "------------------------------------------------------------"
echo "Step 2: Extract patch embeddings (GigaPath / UNIv2 / CONCH / STPath)"
echo "------------------------------------------------------------"
python "${OUTPUT_DIR}/extract_embeddings.py" \
    --patch_dir        "${PATCH_DIR}" \
    --emb_dir          "${EMB_DIR}" \
    --stpath_root      "${STPATH_ROOT}" \
    --gpu              "${GPU}" \
    --batch_size       16 \
    --stpath_batch_size 256

echo ""
echo "------------------------------------------------------------"
echo "Step 3: mIF inference"
echo "------------------------------------------------------------"
python "${OUTPUT_DIR}/inference.py" \
    --patch_dir        "${PATCH_DIR}" \
    --emb_dir          "${EMB_DIR}" \
    --checkpoint       "${CHECKPOINT}" \
    --gigatime_weights "${GIGATIME_WEIGHTS}" \
    --output           "${RESULT_FILE}" \
    --gpu              "${GPU}"

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Results: ${RESULT_FILE}"
echo "============================================================"
