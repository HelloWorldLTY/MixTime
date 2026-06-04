"""
Training script for Mixed HEMIT + Orion Dataset Training
Uses 17-channel model with channel masking for HEMIT (3-channel) samples

HEMIT to Orion channel mapping:
- DAPI (idx 2) -> Hoechst (idx 0)
- CD3 (idx 1) -> CD3e (idx 10)
- panCK (idx 0) -> Pan-CK (idx 15)
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.core.composition import Compose

from scipy.stats import pearsonr, spearmanr

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")

try:
    import tifffile as tiff
except ImportError:
    raise ImportError("tifffile is required. Install with: pip install tifffile")

from models.multi_expert_v2 import create_multi_expert_v2
from losses.channel_focal import get_loss_function, MaskedMixPearsonLoss


# ============================================================================
# Configuration
# ============================================================================

# Orion dataset channel names (17 channels)
ORION_CHANNEL_NAMES = [
    'Hoechst', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
    'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]
OUT_CHANNELS = 17

# HEMIT to Orion channel mapping: HEMIT_idx -> Orion_idx
# HEMIT channels: [panCK(0), CD3(1), DAPI(2)]
# Orion channels: [Hoechst(0), ..., CD3e(10), ..., Pan-CK(15), ...]
HEMIT_TO_ORION = {
    0: 15,  # panCK -> Pan-CK
    1: 10,  # CD3 -> CD3e
    2: 0,   # DAPI -> Hoechst
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Expert Fusion Model on Mixed HEMIT+Orion Dataset')

    # Data paths
    parser.add_argument('--hemit_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/HEMIT',
                        help='Path to HEMIT dataset')
    parser.add_argument('--orion_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/ORION/ORIONCRC_dataset_tile_20x',
                        help='Path to Orion dataset')
    parser.add_argument('--emb_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME',
                        help='Path to embedding directory')
    parser.add_argument('--gigatime_weights', type=str,
                        default=None,
                        help='Path to pretrained GigaTIME weights')
    parser.add_argument('--npy_dir', type=str, default=None,
                        help='Directory with preprocessed .npy labels for Orion (faster loading)')

    # Model architecture settings
    parser.add_argument('--freeze_gigatime', type=lambda x: x.lower() == 'true',
                        default=True, help='Freeze GigaTIME backbone')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_tokens', type=int, default=8,
                        help='Number of embedding tokens')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Configurable model components
    parser.add_argument('--use_cross_expert', action='store_true',
                        help='Enable cross-expert interaction')
    parser.add_argument('--use_dynamic_gating', action='store_true',
                        help='Enable dynamic gating (input-dependent expert weights)')
    parser.add_argument('--use_multiscale_film', action='store_true',
                        help='Enable multi-scale FiLM injection')
    parser.add_argument('--stpath_feature_type', type=str, default='emb',
                        choices=['emb', 'gene'],
                        help='STPath feature type: emb (512-dim) or gene (~30k-dim gene expression)')
    parser.add_argument('--disable_stpath', action='store_true',
                        help='Disable STPath expert (use only UNI and CONCH)')
    parser.add_argument('--disable_uni', action='store_true',
                        help='Disable UNI expert')
    parser.add_argument('--disable_conch', action='store_true',
                        help='Disable CONCH expert')

    # Loss settings
    parser.add_argument('--loss_type', type=str, default='masked_mix_pearson',
                        choices=['smooth_l1', 'focal', 'weighted', 'mix_pearson', 'masked_mix_pearson'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma for focal loss')
    parser.add_argument('--channel_weights', type=float, nargs=17,
                        default=[1.0] * 17,
                        help='Channel weights for 17 channels')
    parser.add_argument('--rel_weight_channel', type=float, default=1.0,
                        help='Weight for channel correlation loss (for mix_pearson)')
    parser.add_argument('--rel_weight_pixel', type=float, default=1.0,
                        help='Weight for pixel correlation loss (for mix_pearson)')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0 recommended for large embeddings)')

    # Output settings
    parser.add_argument('--output_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME/outputs',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default='mixed_hemit_orion',
                        help='Experiment name')

    # Wandb settings
    parser.add_argument('--wandb_project', type=str, default='gigatime-mixed',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # Checkpoint settings
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model (load weights only, restart training from epoch 1)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Dataset Classes
# ============================================================================

class HEMITDataset(Dataset):
    """HEMIT Dataset that outputs 17-channel format with channel mask"""

    def __init__(
        self,
        csv_path,
        data_dir,
        emb_uni_path,
        emb_conch_path,
        emb_stpath_path,
        transform=None,
        input_size=512,
        stpath_feature_type='emb',
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.input_size = input_size
        self.stpath_feature_type = stpath_feature_type

        # Determine split from csv_path (train/val/test)
        if 'train' in csv_path:
            self.split = 'train'
        elif 'val' in csv_path:
            self.split = 'val'
        else:
            self.split = 'test'

        # Load embeddings
        print(f"Loading HEMIT {self.split} embeddings...")
        self.emb_uni = torch.load(emb_uni_path, weights_only=False)
        self.emb_conch = torch.load(emb_conch_path, weights_only=False)
        emb_stpath_data = pd.read_pickle(emb_stpath_path)

        # Choose STPath feature type
        if stpath_feature_type == 'gene':
            assert 'pred' in emb_stpath_data, f"Missing 'pred' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['pred']
            print(f"  Using STPath gene expression predictions")
        else:
            assert 'emb' in emb_stpath_data, f"Missing 'emb' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['emb']
            print(f"  Using STPath embedding features")

        if isinstance(emb_stpath_raw, np.ndarray):
            self.emb_stpath = torch.from_numpy(emb_stpath_raw)
        else:
            self.emb_stpath = emb_stpath_raw

        # Validate data alignment
        n_samples = len(self.df)
        assert self.emb_uni.shape[0] == n_samples, \
            f"UNI embedding size {self.emb_uni.shape[0]} != CSV size {n_samples}"
        assert self.emb_conch.shape[0] == n_samples, \
            f"CONCH embedding size {self.emb_conch.shape[0]} != CSV size {n_samples}"
        assert self.emb_stpath.shape[0] == n_samples, \
            f"STPath embedding size {self.emb_stpath.shape[0]} != CSV size {n_samples}"

        print(f"HEMIT {self.split} dataset size: {len(self.df)}")
        print(f"  UNI: {self.emb_uni.shape}, CONCH: {self.emb_conch.shape}, STPath: {self.emb_stpath.shape}")

        # Normalization params
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Create channel mask for HEMIT (only 3 channels valid in 17-channel space)
        self.channel_mask = torch.zeros(OUT_CHANNELS, dtype=torch.float32)
        for hemit_idx, orion_idx in HEMIT_TO_ORION.items():
            self.channel_mask[orion_idx] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['name']

        # HEMIT path format: he/{split}/input/{name}.tif, he/{split}/label/{name}.tif
        img_path = os.path.join(self.data_dir, 'he', self.split, 'input', f'{name}.tif')
        label_path = os.path.join(self.data_dir, 'he', self.split, 'label', f'{name}.tif')

        # Load H&E image
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).copy()

        # Load 3-channel IF label
        label_3ch = tiff.imread(label_path)
        label_3ch = np.asarray(label_3ch).copy()

        # Handle dimension format: ensure [H, W, C]
        if label_3ch.shape[0] == 3 and len(label_3ch.shape) == 3:
            label_3ch = label_3ch.transpose(1, 2, 0)

        # Create 17-channel label with zeros
        H, W = label_3ch.shape[:2]
        label_17ch = np.zeros((H, W, OUT_CHANNELS), dtype=np.float32)

        # Map HEMIT channels to Orion channels
        # HEMIT: [panCK(0), CD3(1), DAPI(2)] -> Orion: [Pan-CK(15), CD3e(10), Hoechst(0)]
        for hemit_idx, orion_idx in HEMIT_TO_ORION.items():
            label_17ch[:, :, orion_idx] = label_3ch[:, :, hemit_idx].astype(np.float32)

        # Apply transforms (resize to 512x512)
        if self.transform:
            augmented = self.transform(image=img, mask=label_17ch)
            img = augmented['image']
            label_17ch = augmented['mask']

        # Convert to tensor format
        img = img.astype('float32') / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        label_17ch = label_17ch.transpose(2, 0, 1)  # HWC -> CHW [17, H, W]

        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(label_17ch).float(),
            'emb_uni': self.emb_uni[idx].float(),
            'emb_conch': self.emb_conch[idx].float(),
            'emb_stpath': self.emb_stpath[idx].float(),
            'channel_mask': self.channel_mask.clone(),  # [17]
            'name': name,
            'source': 'hemit'
        }


class OrionDataset(Dataset):
    """Orion Dataset with embeddings for 17-channel IF prediction"""

    def __init__(
        self,
        csv_path,
        data_dir,
        emb_uni_path,
        emb_conch_path,
        emb_stpath_path,
        transform=None,
        input_size=512,
        stpath_feature_type='emb',
        npy_dir=None
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.input_size = input_size
        self.stpath_feature_type = stpath_feature_type
        self.npy_dir = npy_dir

        # Load embeddings
        print(f"Loading Orion embeddings...")
        self.emb_uni = torch.load(emb_uni_path, weights_only=False)
        self.emb_conch = torch.load(emb_conch_path, weights_only=False)
        emb_stpath_data = pd.read_pickle(emb_stpath_path)

        # Choose STPath feature type
        if stpath_feature_type == 'gene':
            assert 'pred' in emb_stpath_data, f"Missing 'pred' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['pred']
            print(f"  Using STPath gene expression predictions (38984-dim)")
        else:
            assert 'emb' in emb_stpath_data, f"Missing 'emb' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['emb']
            print(f"  Using STPath embedding features")

        if isinstance(emb_stpath_raw, np.ndarray):
            self.emb_stpath = torch.from_numpy(emb_stpath_raw)
        else:
            self.emb_stpath = emb_stpath_raw

        # Validate data alignment
        n_samples = len(self.df)
        assert self.emb_uni.shape[0] == n_samples, \
            f"UNI embedding size {self.emb_uni.shape[0]} != CSV size {n_samples}"
        assert self.emb_conch.shape[0] == n_samples, \
            f"CONCH embedding size {self.emb_conch.shape[0]} != CSV size {n_samples}"
        assert self.emb_stpath.shape[0] == n_samples, \
            f"STPath embedding size {self.emb_stpath.shape[0]} != CSV size {n_samples}"

        print(f"Orion dataset size: {len(self.df)}")
        print(f"  UNI: {self.emb_uni.shape}, CONCH: {self.emb_conch.shape}, STPath: {self.emb_stpath.shape}")

        # Normalization params
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Full channel mask (all 17 channels valid)
        self.channel_mask = torch.ones(OUT_CHANNELS, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Orion path format: image_path (he/xxx.jpeg), target_path (if/xxx.tiff)
        img_path = os.path.join(self.data_dir, row['image_path'])
        label_path = os.path.join(self.data_dir, row['target_path'])

        # Load H&E image (JPEG)
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).copy()

        # Load label - try .npy first (faster), fallback to TIFF
        if self.npy_dir:
            npy_path = os.path.join(self.npy_dir, row['target_path'].replace('.tiff', '.npy'))
            if os.path.exists(npy_path):
                label = np.load(npy_path)
            else:
                label = tiff.imread(label_path)
        else:
            label = tiff.imread(label_path)
        label = np.asarray(label).copy()

        # Handle different TIFF dimension formats
        if label.shape[0] == OUT_CHANNELS and len(label.shape) == 3:
            # [C, H, W] format -> [H, W, C]
            label = label.transpose(1, 2, 0)

        # Apply transforms (resize to 512x512)
        if self.transform:
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        # Convert to tensor format
        img = img.astype('float32') / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        label = label.astype('float32')
        label = label.transpose(2, 0, 1)  # HWC -> CHW [17, H, W]

        # Get sample name from image path
        name = os.path.basename(row['image_path']).replace('.jpeg', '')

        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(label).float(),
            'emb_uni': self.emb_uni[idx].float(),
            'emb_conch': self.emb_conch[idx].float(),
            'emb_stpath': self.emb_stpath[idx].float(),
            'channel_mask': self.channel_mask.clone(),  # [17], all ones
            'name': name,
            'source': 'orion'
        }


def get_transforms(input_size, is_train=True):
    """Get data transforms"""
    if is_train:
        return Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        return Compose([
            A.Resize(input_size, input_size),
        ])


def create_dataloaders(args):
    """Create train, val, test dataloaders for mixed HEMIT+Orion dataset"""

    # ========== Orion Paths ==========
    orion_train_csv = os.path.join(args.orion_dir, 'train_dataframe.csv')
    orion_val_csv = os.path.join(args.orion_dir, 'val_dataframe.csv')
    orion_test_csv = os.path.join(args.orion_dir, 'test_dataframe.csv')

    # Orion embedding paths
    emb_uni_orion_train = os.path.join(args.emb_dir, 'univ2emb', 'orion_train.pkl')
    emb_uni_orion_val = os.path.join(args.emb_dir, 'univ2emb', 'orion_val.pkl')
    emb_uni_orion_test = os.path.join(args.emb_dir, 'univ2emb', 'orion_test.pkl')

    emb_conch_orion_train = os.path.join(args.emb_dir, 'conchemb', 'orion_train.pkl')
    emb_conch_orion_val = os.path.join(args.emb_dir, 'conchemb', 'orion_val.pkl')
    emb_conch_orion_test = os.path.join(args.emb_dir, 'conchemb', 'orion_test.pkl')

    emb_stpath_orion_train = os.path.join(args.emb_dir, 'stpathinfo', 'orion_train.pkl')
    emb_stpath_orion_val = os.path.join(args.emb_dir, 'stpathinfo', 'orion_val.pkl')
    emb_stpath_orion_test = os.path.join(args.emb_dir, 'stpathinfo', 'orion_test.pkl')

    # ========== HEMIT Paths ==========
    hemit_train_csv = os.path.join(args.hemit_dir, 'train_data.csv')
    hemit_val_csv = os.path.join(args.hemit_dir, 'val_data.csv')
    hemit_test_csv = os.path.join(args.hemit_dir, 'test_data.csv')

    # HEMIT embedding paths (named 'meiphi' in the embedding files)
    emb_uni_hemit_train = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_train.pkl')
    emb_uni_hemit_val = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_val.pkl')
    emb_uni_hemit_test = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_test.pkl')

    emb_conch_hemit_train = os.path.join(args.emb_dir, 'conchemb', 'meiphi_train.pkl')
    emb_conch_hemit_val = os.path.join(args.emb_dir, 'conchemb', 'meiphi_val.pkl')
    emb_conch_hemit_test = os.path.join(args.emb_dir, 'conchemb', 'meiphi_test.pkl')

    emb_stpath_hemit_train = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_train.pkl')
    emb_stpath_hemit_val = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_val.pkl')
    emb_stpath_hemit_test = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_test.pkl')

    # ========== Create Datasets ==========
    print("\n" + "="*60)
    print("Creating Orion datasets...")
    print("="*60)

    orion_train = OrionDataset(
        csv_path=orion_train_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_orion_train,
        emb_conch_path=emb_conch_orion_train,
        emb_stpath_path=emb_stpath_orion_train,
        transform=get_transforms(512, is_train=True),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
    )

    orion_val = OrionDataset(
        csv_path=orion_val_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_orion_val,
        emb_conch_path=emb_conch_orion_val,
        emb_stpath_path=emb_stpath_orion_val,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
    )

    orion_test = OrionDataset(
        csv_path=orion_test_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_orion_test,
        emb_conch_path=emb_conch_orion_test,
        emb_stpath_path=emb_stpath_orion_test,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
    )

    # Check if HEMIT embeddings exist
    hemit_embeddings_exist = (
        os.path.exists(emb_uni_hemit_train) and
        os.path.exists(emb_conch_hemit_train) and
        os.path.exists(emb_stpath_hemit_train)
    )

    if hemit_embeddings_exist:
        print("\n" + "="*60)
        print("Creating HEMIT datasets...")
        print("="*60)

        hemit_train = HEMITDataset(
            csv_path=hemit_train_csv,
            data_dir=args.hemit_dir,
            emb_uni_path=emb_uni_hemit_train,
            emb_conch_path=emb_conch_hemit_train,
            emb_stpath_path=emb_stpath_hemit_train,
            transform=get_transforms(512, is_train=True),
            stpath_feature_type=args.stpath_feature_type,
        )

        hemit_val = HEMITDataset(
            csv_path=hemit_val_csv,
            data_dir=args.hemit_dir,
            emb_uni_path=emb_uni_hemit_val,
            emb_conch_path=emb_conch_hemit_val,
            emb_stpath_path=emb_stpath_hemit_val,
            transform=get_transforms(512, is_train=False),
            stpath_feature_type=args.stpath_feature_type,
        )

        hemit_test = HEMITDataset(
            csv_path=hemit_test_csv,
            data_dir=args.hemit_dir,
            emb_uni_path=emb_uni_hemit_test,
            emb_conch_path=emb_conch_hemit_test,
            emb_stpath_path=emb_stpath_hemit_test,
            transform=get_transforms(512, is_train=False),
            stpath_feature_type=args.stpath_feature_type,
        )

        # Combine datasets
        print("\n" + "="*60)
        print("Combining datasets...")
        print("="*60)
        train_dataset = ConcatDataset([orion_train, hemit_train])
        val_dataset = ConcatDataset([orion_val, hemit_val])
        test_dataset = ConcatDataset([orion_test, hemit_test])

        print(f"Combined train: {len(train_dataset)} (Orion: {len(orion_train)}, HEMIT: {len(hemit_train)})")
        print(f"Combined val: {len(val_dataset)} (Orion: {len(orion_val)}, HEMIT: {len(hemit_val)})")
        print(f"Combined test: {len(test_dataset)} (Orion: {len(orion_test)}, HEMIT: {len(hemit_test)})")
    else:
        print("\n" + "="*60)
        print("WARNING: HEMIT embeddings not found. Using Orion only.")
        print(f"Expected: {emb_uni_hemit_train}")
        print("="*60)
        train_dataset = orion_train
        val_dataset = orion_val
        test_dataset = orion_test

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Metrics
# ============================================================================

def calculate_correlations(pred, target, channel_mask=None):
    """Calculate Pearson and Spearman correlations per channel"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    B, C, H, W = pred.shape
    pcc_list = []
    scc_list = []

    for c in range(C):
        pcc_channel = []
        scc_channel = []

        for b in range(B):
            # Skip if this channel is masked out
            if channel_mask is not None:
                mask_b = channel_mask[b] if channel_mask.dim() == 2 else channel_mask
                if mask_b[c].item() == 0:
                    continue

            p = pred[b, c].flatten()
            t = target[b, c].flatten()

            valid = ~(np.isnan(p) | np.isnan(t))
            p, t = p[valid], t[valid]

            if len(p) > 0:
                pcc, _ = pearsonr(p, t)
                scc, _ = spearmanr(p, t)
                pcc_channel.append(pcc)
                scc_channel.append(scc)

        if len(pcc_channel) > 0:
            pcc_list.append(np.nanmean(pcc_channel))
            scc_list.append(np.nanmean(scc_channel))
        else:
            pcc_list.append(np.nan)
            scc_list.append(np.nan)

    return np.array(pcc_list), np.array(scc_list)


def format_channel_metrics(pcc, scc, channel_names):
    """Format channel metrics for printing"""
    metrics = {}
    for i, name in enumerate(channel_names):
        metrics[f'pcc_{name}'] = pcc[i]
        metrics[f'scc_{name}'] = scc[i]
    metrics['pcc_mean'] = np.nanmean(pcc)
    metrics['scc_mean'] = np.nanmean(scc)
    return metrics


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_stpath=True, scaler=None):
    """Train for one epoch with optional AMP support"""
    model.train()
    use_amp = scaler is not None
    use_masked_loss = isinstance(criterion, MaskedMixPearsonLoss)

    total_loss = 0
    expert_weights_sum = None

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        emb_uni = batch['emb_uni'].to(device)
        emb_conch = batch['emb_conch'].to(device)
        emb_stpath = batch['emb_stpath'].to(device) if use_stpath else None
        channel_mask = batch['channel_mask'].to(device) if 'channel_mask' in batch else None

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            pred, info = model(images, emb_uni, emb_conch, emb_stpath)

            # Compute loss with channel mask if using masked loss
            if use_masked_loss and channel_mask is not None:
                loss = criterion(pred, labels, channel_mask)
            else:
                loss = criterion(pred, labels)

        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track expert weights
        if 'expert_weights' in info:
            w = info['expert_weights']
            if w.dim() == 1:
                w = w.detach().cpu().numpy()
            else:
                w = w.mean(dim=0).detach().cpu().numpy()
            if expert_weights_sum is None:
                expert_weights_sum = w
            else:
                expert_weights_sum += w

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    # Average metrics
    avg_loss = total_loss / len(train_loader)
    avg_pcc = np.zeros(len(ORION_CHANNEL_NAMES))
    avg_scc = np.zeros(len(ORION_CHANNEL_NAMES))

    avg_expert_weights = None
    if expert_weights_sum is not None:
        avg_expert_weights = expert_weights_sum / len(train_loader)

    metrics = format_channel_metrics(avg_pcc, avg_scc, ORION_CHANNEL_NAMES)
    metrics['loss'] = avg_loss
    metrics['expert_weights'] = avg_expert_weights

    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, phase='Val', use_stpath=True, use_amp=True, compute_metrics=True):
    """Evaluate model with optional AMP support. Reports loss and metrics per source (orion/hemit)."""
    model.eval()
    use_masked_loss = isinstance(criterion, MaskedMixPearsonLoss)

    # HEMIT valid channel indices in 17-channel space
    HEMIT_CHANNELS = [0, 10, 15]  # Hoechst, CD3e, Pan-CK

    # Track loss and metrics per source
    total_loss = 0
    orion_loss_sum = 0
    hemit_loss_sum = 0
    orion_count = 0
    hemit_count = 0

    # Track PCC/SCC per source
    orion_pcc_list = []
    orion_scc_list = []
    hemit_pcc_list = []  # Only 3 channels
    hemit_scc_list = []

    pbar = tqdm(data_loader, desc=f'[{phase}]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        emb_uni = batch['emb_uni'].to(device)
        emb_conch = batch['emb_conch'].to(device)
        emb_stpath = batch['emb_stpath'].to(device) if use_stpath else None
        channel_mask = batch['channel_mask'].to(device) if 'channel_mask' in batch else None
        sources = batch.get('source', None)  # List of 'orion' or 'hemit'

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            pred, _ = model(images, emb_uni, emb_conch, emb_stpath)

            if use_masked_loss and channel_mask is not None:
                loss = criterion(pred, labels, channel_mask)
            else:
                loss = criterion(pred, labels)

            # Compute per-sample loss for source-wise tracking
            if sources is not None:
                B = pred.shape[0]
                per_sample_loss = F.smooth_l1_loss(pred, labels, reduction='none').mean(dim=(1, 2, 3))
                for i in range(B):
                    sample_loss = per_sample_loss[i].item()
                    if sources[i] == 'orion':
                        orion_loss_sum += sample_loss
                        orion_count += 1
                    else:  # hemit
                        hemit_loss_sum += sample_loss
                        hemit_count += 1

        # Compute metrics only if requested - separate by source
        if compute_metrics and sources is not None:
            pred_np = pred.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            B = pred_np.shape[0]

            for i in range(B):
                if sources[i] == 'orion':
                    # Orion: compute all 17 channels
                    pcc_sample = []
                    scc_sample = []
                    for c in range(17):
                        p = pred_np[i, c].flatten()
                        t = labels_np[i, c].flatten()
                        valid = ~(np.isnan(p) | np.isnan(t))
                        p, t = p[valid], t[valid]
                        if len(p) > 0:
                            pcc, _ = pearsonr(p, t)
                            scc, _ = spearmanr(p, t)
                            pcc_sample.append(pcc)
                            scc_sample.append(scc)
                        else:
                            pcc_sample.append(np.nan)
                            scc_sample.append(np.nan)
                    orion_pcc_list.append(pcc_sample)
                    orion_scc_list.append(scc_sample)
                else:
                    # HEMIT: only compute 3 valid channels
                    pcc_sample = []
                    scc_sample = []
                    for c in HEMIT_CHANNELS:
                        p = pred_np[i, c].flatten()
                        t = labels_np[i, c].flatten()
                        valid = ~(np.isnan(p) | np.isnan(t))
                        p, t = p[valid], t[valid]
                        if len(p) > 0:
                            pcc, _ = pearsonr(p, t)
                            scc, _ = spearmanr(p, t)
                            pcc_sample.append(pcc)
                            scc_sample.append(scc)
                        else:
                            pcc_sample.append(np.nan)
                            scc_sample.append(np.nan)
                    hemit_pcc_list.append(pcc_sample)
                    hemit_scc_list.append(scc_sample)

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    # Build metrics dict
    metrics = {}

    if compute_metrics:
        # Orion metrics (17 channels)
        if len(orion_pcc_list) > 0:
            orion_pcc = np.nanmean(orion_pcc_list, axis=0)
            orion_scc = np.nanmean(orion_scc_list, axis=0)
            for i, name in enumerate(ORION_CHANNEL_NAMES):
                metrics[f'orion_pcc_{name}'] = orion_pcc[i]
                metrics[f'orion_scc_{name}'] = orion_scc[i]
            metrics['orion_pcc_mean'] = np.nanmean(orion_pcc)
            metrics['orion_scc_mean'] = np.nanmean(orion_scc)

        # HEMIT metrics (3 channels: Hoechst, CD3e, Pan-CK)
        if len(hemit_pcc_list) > 0:
            hemit_pcc = np.nanmean(hemit_pcc_list, axis=0)
            hemit_scc = np.nanmean(hemit_scc_list, axis=0)
            hemit_channel_names = ['Hoechst', 'CD3e', 'Pan-CK']
            for i, name in enumerate(hemit_channel_names):
                metrics[f'hemit_pcc_{name}'] = hemit_pcc[i]
                metrics[f'hemit_scc_{name}'] = hemit_scc[i]
            metrics['hemit_pcc_mean'] = np.nanmean(hemit_pcc)
            metrics['hemit_scc_mean'] = np.nanmean(hemit_scc)

        # Combined metrics (use orion as primary since it has all channels)
        if len(orion_pcc_list) > 0:
            metrics['pcc_mean'] = metrics['orion_pcc_mean']
            metrics['scc_mean'] = metrics['orion_scc_mean']
        else:
            metrics['pcc_mean'] = metrics.get('hemit_pcc_mean', 0)
            metrics['scc_mean'] = metrics.get('hemit_scc_mean', 0)

    metrics['loss'] = avg_loss
    metrics['orion_loss'] = orion_loss_sum / orion_count if orion_count > 0 else 0
    metrics['hemit_loss'] = hemit_loss_sum / hemit_count if hemit_count > 0 else 0
    metrics['orion_count'] = orion_count
    metrics['hemit_count'] = hemit_count

    return metrics


def train(args):
    """Main training function"""
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_name = args.exp_name

    exp_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Output directory: {exp_dir}")

    # Initialize wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # Download GigaTIME weights if needed
    if args.gigatime_weights is None:
        from huggingface_hub import snapshot_download
        print("Downloading GigaTIME weights from HuggingFace...")
        repo_id = "prov-gigatime/GigaTIME"
        local_dir = snapshot_download(repo_id=repo_id)
        args.gigatime_weights = os.path.join(local_dir, "model.pth")

    # Get embedding dimensions
    if args.disable_uni:
        uni_dim = 0
        print("  UNI: DISABLED")
    else:
        uni_dim = 1536
        print(f"  UNI feature dim: {uni_dim}")

    if args.disable_conch:
        conch_dim = 0
        print("  CONCH: DISABLED")
    else:
        conch_dim = 768
        print(f"  CONCH feature dim: {conch_dim}")

    if args.disable_stpath:
        stpath_dim = 0
        print("  STPath: DISABLED")
    else:
        # Get stpath_dim from first dataset in train_loader
        sample_batch = next(iter(train_loader))
        stpath_dim = sample_batch['emb_stpath'].shape[1]
        print(f"  STPath feature dim: {stpath_dim}")

    num_active_experts = sum([uni_dim > 0, conch_dim > 0, stpath_dim > 0])
    if num_active_experts == 0:
        raise ValueError("At least one expert must be enabled.")
    print(f"  Active experts: {num_active_experts}")

    # Create model
    print("Creating model...")
    print(f"  out_channels: {OUT_CHANNELS}")
    print(f"  use_cross_expert: {args.use_cross_expert}")
    print(f"  use_dynamic_gating: {args.use_dynamic_gating}")
    print(f"  use_multiscale_film: {args.use_multiscale_film}")
    print(f"  stpath_feature_type: {args.stpath_feature_type}")

    model = create_multi_expert_v2(
        weights_path=args.gigatime_weights,
        freeze_gigatime=args.freeze_gigatime,
        num_heads=args.num_heads,
        num_tokens=args.num_tokens,
        dropout=args.dropout,
        use_cross_expert=args.use_cross_expert,
        use_dynamic_gating=args.use_dynamic_gating,
        use_multiscale_film=args.use_multiscale_film,
        uni_dim=uni_dim,
        conch_dim=conch_dim,
        stpath_dim=stpath_dim,
        out_channels=OUT_CHANNELS
    )

    # Load pretrained model weights (only model, not optimizer/scheduler)
    if args.pretrained_model:
        print(f"\nLoading pretrained model from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_loss' in checkpoint:
                print(f"  Pretrained val_loss: {checkpoint['val_loss']:.4f}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("  Loaded model weights (direct state dict)")
        print("  NOTE: Optimizer/scheduler will be re-initialized, training starts from epoch 1")

    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    print(f"Loss type: {args.loss_type}")
    criterion = get_loss_function(
        loss_type=args.loss_type,
        gamma=args.focal_gamma,
        channel_weights=args.channel_weights,
        rel_weight_channel=args.rel_weight_channel,
        rel_weight_pixel=args.rel_weight_pixel
    )

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    use_stpath = not args.disable_stpath
    scaler = GradScaler()
    use_amp = True
    print(f"AMP (Automatic Mixed Precision): Enabled")

    # Training loop
    best_val_loss = float('inf')
    best_val_pcc = 0
    patience_counter = 0
    start_epoch = 1

    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        if 'best_val_pcc' in checkpoint:
            best_val_pcc = checkpoint['best_val_pcc']
        if 'patience_counter' in checkpoint:
            patience_counter = checkpoint['patience_counter']
        print(f"  Resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_stpath, scaler
        )

        scheduler.step()

        log_dict = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train/loss': train_metrics['loss'],
        }

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}")

        # Validate
        compute_full_metrics = (epoch % args.val_every == 0)
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val', use_stpath, use_amp,
                               compute_metrics=compute_full_metrics)
        log_dict['val/loss'] = val_metrics['loss']
        log_dict['val/orion_loss'] = val_metrics['orion_loss']
        log_dict['val/hemit_loss'] = val_metrics['hemit_loss']

        if compute_full_metrics:
            # Log Orion metrics (17 channels)
            if 'orion_pcc_mean' in val_metrics:
                log_dict['val/orion_pcc_mean'] = val_metrics['orion_pcc_mean']
                log_dict['val/orion_scc_mean'] = val_metrics['orion_scc_mean']
                for name in ORION_CHANNEL_NAMES:
                    log_dict[f'val/orion_pcc_{name}'] = val_metrics.get(f'orion_pcc_{name}', 0)

            # Log HEMIT metrics (3 channels)
            if 'hemit_pcc_mean' in val_metrics:
                log_dict['val/hemit_pcc_mean'] = val_metrics['hemit_pcc_mean']
                log_dict['val/hemit_scc_mean'] = val_metrics['hemit_scc_mean']
                for name in ['Hoechst', 'CD3e', 'Pan-CK']:
                    log_dict[f'val/hemit_pcc_{name}'] = val_metrics.get(f'hemit_pcc_{name}', 0)

            # Print summary
            orion_pcc = val_metrics.get('orion_pcc_mean', 0)
            hemit_pcc = val_metrics.get('hemit_pcc_mean', 0)
            print(f"\nVal - Loss: {val_metrics['loss']:.4f} (Orion: {val_metrics['orion_loss']:.4f}, HEMIT: {val_metrics['hemit_loss']:.4f})")
            print(f"  Orion PCC: {orion_pcc:.4f} | HEMIT PCC: {hemit_pcc:.4f}")
            # Print key channels for both
            if 'orion_pcc_Hoechst' in val_metrics:
                print(f"  Orion - Hoechst: {val_metrics['orion_pcc_Hoechst']:.4f}, "
                      f"CD3e: {val_metrics['orion_pcc_CD3e']:.4f}, "
                      f"Pan-CK: {val_metrics['orion_pcc_Pan-CK']:.4f}")
            if 'hemit_pcc_Hoechst' in val_metrics:
                print(f"  HEMIT - Hoechst: {val_metrics['hemit_pcc_Hoechst']:.4f}, "
                      f"CD3e: {val_metrics['hemit_pcc_CD3e']:.4f}, "
                      f"Pan-CK: {val_metrics['hemit_pcc_Pan-CK']:.4f}")
        else:
            print(f"\nVal - Loss: {val_metrics['loss']:.4f} (Orion: {val_metrics['orion_loss']:.4f}, HEMIT: {val_metrics['hemit_loss']:.4f})")

        # Also compute test loss (no PCC/SCC to save time)
        test_metrics = evaluate(model, test_loader, criterion, device, 'Test', use_stpath, use_amp,
                                compute_metrics=False)
        log_dict['test/loss'] = test_metrics['loss']
        log_dict['test/orion_loss'] = test_metrics['orion_loss']
        log_dict['test/hemit_loss'] = test_metrics['hemit_loss']
        print(f"Test - Loss: {test_metrics['loss']:.4f} (Orion: {test_metrics['orion_loss']:.4f}, HEMIT: {test_metrics['hemit_loss']:.4f})")

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if compute_full_metrics:
                best_val_pcc = val_metrics['pcc_mean']
            patience_counter = 0

            save_path = os.path.join(exp_dir, 'best_model.pth')
            model_to_save = model.module if hasattr(model, 'module') else model
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': vars(args)
            }
            if compute_full_metrics:
                save_dict['val_pcc_mean'] = val_metrics['pcc_mean']
            torch.save(save_dict, save_path)
            print(f"Saved best model (val_loss={val_metrics['loss']:.4f})")
        else:
            patience_counter += 1

        # Save checkpoint
        ckpt_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pth')
        model_to_save = model.module if hasattr(model, 'module') else model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'patience_counter': patience_counter,
            'best_val_loss': best_val_loss,
            'best_val_pcc': best_val_pcc,
            'config': vars(args)
        }
        torch.save(ckpt_dict, ckpt_path)

        # Delete old checkpoint (keep latest 10)
        old_ckpt = os.path.join(exp_dir, f'checkpoint_epoch_{epoch - 10}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Final test
    print("\n" + "="*60)
    print("Final test with best model...")
    print("="*60)

    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', use_stpath, use_amp)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    if 'orion_pcc_mean' in test_metrics:
        print(f"  Orion PCC Mean: {test_metrics['orion_pcc_mean']:.4f}")
        print(f"  Orion SCC Mean: {test_metrics['orion_scc_mean']:.4f}")
    if 'hemit_pcc_mean' in test_metrics:
        print(f"  HEMIT PCC Mean: {test_metrics['hemit_pcc_mean']:.4f}")
        print(f"  HEMIT SCC Mean: {test_metrics['hemit_scc_mean']:.4f}")
    print(f"\nOrion Per-channel PCC:")
    for name in ORION_CHANNEL_NAMES:
        pcc_key = f'orion_pcc_{name}'
        if pcc_key in test_metrics:
            print(f"  {name}: {test_metrics[pcc_key]:.4f}")

    if not args.no_wandb and WANDB_AVAILABLE:
        final_log = {
            'final_test/loss': test_metrics['loss'],
        }
        if 'orion_pcc_mean' in test_metrics:
            final_log['final_test/orion_pcc_mean'] = test_metrics['orion_pcc_mean']
            final_log['final_test/orion_scc_mean'] = test_metrics['orion_scc_mean']
            for name in ORION_CHANNEL_NAMES:
                final_log[f'final_test/orion_pcc_{name}'] = test_metrics.get(f'orion_pcc_{name}', 0)
        if 'hemit_pcc_mean' in test_metrics:
            final_log['final_test/hemit_pcc_mean'] = test_metrics['hemit_pcc_mean']
            final_log['final_test/hemit_scc_mean'] = test_metrics['hemit_scc_mean']
        wandb.log(final_log)
        wandb.finish()

    # Save results
    import json
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_pcc': best_val_pcc,
        'test_loss': test_metrics['loss'],
        'test_orion_loss': test_metrics.get('orion_loss', 0),
        'test_hemit_loss': test_metrics.get('hemit_loss', 0),
        'config': {
            'use_cross_expert': args.use_cross_expert,
            'use_dynamic_gating': args.use_dynamic_gating,
            'use_multiscale_film': args.use_multiscale_film,
            'loss_type': args.loss_type,
            'out_channels': OUT_CHANNELS,
        }
    }
    # Orion metrics
    if 'orion_pcc_mean' in test_metrics:
        results['test_orion_pcc_mean'] = test_metrics['orion_pcc_mean']
        results['test_orion_scc_mean'] = test_metrics['orion_scc_mean']
        for name in ORION_CHANNEL_NAMES:
            results[f'test_orion_pcc_{name}'] = test_metrics.get(f'orion_pcc_{name}', 0)
            results[f'test_orion_scc_{name}'] = test_metrics.get(f'orion_scc_{name}', 0)
    # HEMIT metrics
    if 'hemit_pcc_mean' in test_metrics:
        results['test_hemit_pcc_mean'] = test_metrics['hemit_pcc_mean']
        results['test_hemit_scc_mean'] = test_metrics['hemit_scc_mean']
        for name in ['Hoechst', 'CD3e', 'Pan-CK']:
            results[f'test_hemit_pcc_{name}'] = test_metrics.get(f'hemit_pcc_{name}', 0)
            results[f'test_hemit_scc_{name}'] = test_metrics.get(f'hemit_scc_{name}', 0)

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {exp_dir}")

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
