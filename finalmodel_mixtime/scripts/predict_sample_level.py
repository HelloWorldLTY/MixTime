"""
Predict 17-channel biomarker expression from slide_report H&E images using trained Orion model.
Outputs ranked biomarker predictions per image.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.multi_expert_v2 import create_multi_expert_v2

# Orion 17 biomarker channel names
ORION_CHANNEL_NAMES = [
    'Hoechst', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
    'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def load_model(checkpoint_path, gigatime_weights, device):
    """Load trained Orion model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # Determine embedding dimensions from config
    uni_dim = 0 if cfg.get('disable_uni', False) else 1536
    conch_dim = 0 if cfg.get('disable_conch', False) else 768
    # stpath_dim must match training; for 'gene' type it's 38984
    if cfg.get('disable_stpath', False):
        stpath_dim = 0
    else:
        # Infer from checkpoint weights
        stpath_key = None
        for k in checkpoint['model_state_dict']:
            if 'stpath_expert' in k and 'emb_to_kv.weight' in k:
                stpath_key = k
                break
        if stpath_key is not None:
            # emb_to_kv weight shape: [feat_dim * num_tokens * 2, emb_dim]
            stpath_dim = checkpoint['model_state_dict'][stpath_key].shape[1]
        else:
            stpath_dim = 38984  # fallback for gene type

    model = create_multi_expert_v2(
        weights_path=gigatime_weights,
        freeze_gigatime=cfg.get('freeze_gigatime', True),
        num_heads=cfg.get('num_heads', 4),
        num_tokens=cfg.get('num_tokens', 8),
        dropout=cfg.get('dropout', 0.1),
        use_cross_expert=cfg.get('use_cross_expert', False),
        use_dynamic_gating=cfg.get('use_dynamic_gating', False),
        use_multiscale_film=cfg.get('use_multiscale_film', False),
        uni_dim=uni_dim,
        conch_dim=conch_dim,
        stpath_dim=stpath_dim,
        out_channels=17
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Experts: UNI={uni_dim}, CONCH={conch_dim}, STPath={stpath_dim}")
    return model


def preprocess_image(img_path, input_size=512):
    """Load and preprocess a single H&E image."""
    img = Image.open(img_path).convert('RGB')  # RGBA -> RGB
    img = img.resize((input_size, input_size), Image.BILINEAR)
    img = np.asarray(img).astype('float32') / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img).float().unsqueeze(0)  # [1, 3, H, W]


def load_embeddings(emb_dir, name, device):
    """Load UNI/CONCH/STPath embeddings for a single image."""
    emb_uni = torch.load(os.path.join(emb_dir, f'{name}_univ2.pkl'),
                         map_location=device, weights_only=False)
    emb_uni = emb_uni.squeeze(0)  # [1, 1536] -> [1536]

    emb_conch = torch.load(os.path.join(emb_dir, f'{name}_conch.pkl'),
                           map_location=device, weights_only=False)
    emb_conch = emb_conch.squeeze(0)  # [1, 768] -> [768]

    emb_stpath = torch.load(os.path.join(emb_dir, f'{name}_stpath.pkl'),
                            map_location=device, weights_only=False)
    # stpath is already [38984] 1D tensor

    return emb_uni.unsqueeze(0), emb_conch.unsqueeze(0), emb_stpath.unsqueeze(0)


def predict_single(model, img_tensor, emb_uni, emb_conch, emb_stpath, device):
    """Run model inference on a single image and return mean expression per channel."""
    img_tensor = img_tensor.to(device)
    emb_uni = emb_uni.to(device)
    emb_conch = emb_conch.to(device)
    emb_stpath = emb_stpath.to(device)

    with torch.no_grad():
        pred, _ = model(img_tensor, emb_uni, emb_conch, emb_stpath)
        # pred: [1, 17, H, W]

    mean_expr = pred[0].mean(dim=(1, 2)).cpu().numpy()  # [17]
    return mean_expr


def main():
    parser = argparse.ArgumentParser(description='Predict biomarker expression from slide_report images')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/orion_gene_mixpearson/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--gigatime_weights', type=str,
                        default='/work/nvme/bdxk/zwang92/hf/hub/models--prov-gigatime--GigaTIME/snapshots/e48822b5419308cf918ae920239408d7b33327fa/model.pth',
                        help='Path to pretrained GigaTIME weights')
    parser.add_argument('--slide_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/GigaTIME/slide_report',
                        help='Root directory containing set1-5 folders')
    parser.add_argument('--emb_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/GigaTIME/report_embedding',
                        help='Directory with per-image embedding files')
    parser.add_argument('--output', type=str,
                        default='outputs/slide_report_predictions.json',
                        help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.gigatime_weights, device)

    # Collect all PNG images from set1-5
    all_results = {}
    sets = sorted([d for d in os.listdir(args.slide_dir)
                   if os.path.isdir(os.path.join(args.slide_dir, d))])

    for set_name in sets:
        set_dir = os.path.join(args.slide_dir, set_name)
        png_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.png')])

        if not png_files:
            print(f"\n--- {set_name}: no PNG files, skipping ---")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {set_name} ({len(png_files)} images)")
        print(f"{'='*60}")

        set_results = []
        for png_file in png_files:
            img_path = os.path.join(set_dir, png_file)
            name = png_file.replace('.png', '')

            # Preprocess image
            img_tensor = preprocess_image(img_path)

            # Load embeddings
            emb_uni, emb_conch, emb_stpath = load_embeddings(args.emb_dir, name, device)

            # Predict
            mean_expr = predict_single(model, img_tensor, emb_uni, emb_conch, emb_stpath, device)

            # Rank biomarkers by mean expression (descending)
            ranked_indices = np.argsort(-mean_expr)
            ranked = [(ORION_CHANNEL_NAMES[i], float(mean_expr[i])) for i in ranked_indices]

            # Print results
            print(f"\n=== {set_name}/{name} ===")
            print(f"{'Rank':<6}{'Biomarker':<15}{'Mean Expression':<15}")
            for rank, (bm, val) in enumerate(ranked, 1):
                print(f"{rank:<6}{bm:<15}{val:<15.4f}")

            set_results.append({
                'image': name,
                'ranked_biomarkers': [{'rank': r+1, 'biomarker': bm, 'mean_expression': val}
                                      for r, (bm, val) in enumerate(ranked)]
            })

        all_results[set_name] = set_results

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
