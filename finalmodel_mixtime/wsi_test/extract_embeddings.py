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
                        help="Number of patches per STPath inference chunk (256 or 512 recommended)")
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

        # --- global coord normalisation (same logic as agent._normalize_coords) ---
        coords_t = torch.from_numpy(coords.astype(np.float32)).to(args.gpu)
        coords_t[:, 0] = coords_t[:, 0] - coords_t[:, 0].min()
        coords_t[:, 1] = coords_t[:, 1] - coords_t[:, 1].min()
        coords_norm = rescale_coords(coords_t)   # still on GPU

        feats_gp_np = feats_gp.numpy().astype(np.float32)
        N_patches   = len(coords_norm)
        bs          = args.stpath_batch_size

        # pre-compute per-batch constant tokens
        organ_id = agent.tokenizer.organ_tokenizer.encode("Others", align_first=True)

        all_preds = []
        for start in tqdm(range(0, N_patches, bs), desc="  STPath"):
            end      = min(start + bs, N_patches)
            actual   = end - start

            c_chunk   = coords_norm[start:end]
            f_chunk   = torch.from_numpy(feats_gp_np[start:end]).to(args.gpu)
            masked_ge = agent._generate_masked_ge_tokens(actual).to(args.gpu)
            tech_ids  = agent._generate_pad_tech_tokens(actual).to(args.gpu)
            organ_ids = torch.full((actual,), organ_id, dtype=torch.long, device=args.gpu)
            batch_idx = torch.zeros(actual, dtype=torch.long, device=args.gpu)

            # STPath model requires batch size == bs; pad last chunk if needed
            if actual < bs:
                pad = bs - actual
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
