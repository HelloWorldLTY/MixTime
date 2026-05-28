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
