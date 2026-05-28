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
