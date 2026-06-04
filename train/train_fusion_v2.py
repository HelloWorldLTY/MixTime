"""
Training script for Multi-Expert Fusion V2 Model
Supports configurable components: Focal Loss, Cross-Expert, Dynamic Gating, Multi-scale FiLM
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.core.composition import Compose

from scipy.stats import pearsonr, spearmanr

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")

from models.multi_expert_v2 import create_multi_expert_v2
from losses.channel_focal import get_loss_function, ChannelFocalLoss


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Expert Fusion V2 Model')

    # Data paths
    parser.add_argument('--hemit_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/HEMIT',
                        help='Path to HEMIT dataset')
    parser.add_argument('--emb_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME',
                        help='Path to embedding directory')
    parser.add_argument('--gigatime_weights', type=str,
                        default=None,
                        help='Path to pretrained GigaTIME weights')

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
    parser.add_argument('--loss_type', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'focal', 'weighted', 'mix_pearson'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma for focal loss')
    parser.add_argument('--channel_weights', type=float, nargs=3,
                        default=[1.0, 2.0, 1.0],
                        help='Channel weights [panCK, CD3, DAPI]')
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output settings
    parser.add_argument('--output_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME/outputs',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default='fusion_v2',
                        help='Experiment name')

    # Wandb settings
    parser.add_argument('--wandb_project', type=str, default='gigatime-fusion',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # Checkpoint settings
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset
# ============================================================================

class HEMITDataset(Dataset):
    """HEMIT Dataset with embeddings"""

    def __init__(
        self,
        csv_path,
        image_dir,
        emb_uni_path,
        emb_conch_path,
        emb_stpath_path,
        transform=None,
        input_size=512,
        stpath_feature_type='emb'
    ):
        self.df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.input_size = input_size
        self.stpath_feature_type = stpath_feature_type

        # Load embeddings
        print(f"Loading embeddings...")
        self.emb_uni = torch.load(emb_uni_path, weights_only=False)
        self.emb_conch = torch.load(emb_conch_path, weights_only=False)
        emb_stpath_data = pd.read_pickle(emb_stpath_path)

        # Choose STPath feature type: 'emb' (512-dim) or 'gene' (~38k-dim)
        if stpath_feature_type == 'gene':
            assert 'pred' in emb_stpath_data, f"Missing 'pred' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['pred']
            print(f"  Using STPath gene expression predictions (38984-dim)")
        else:
            assert 'emb' in emb_stpath_data, f"Missing 'emb' key in {emb_stpath_path}"
            emb_stpath_raw = emb_stpath_data['emb']
            print(f"  Using STPath embedding features")

        # Convert to tensor if numpy array
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

        print(f"Dataset size: {len(self.df)}")
        print(f"  UNI: {self.emb_uni.shape}, CONCH: {self.emb_conch.shape}, STPath: {self.emb_stpath.shape}")

        # Normalization params
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['name']

        # Load H&E image
        img_path = os.path.join(self.image_dir, 'input', f'{name}.tif')
        img = Image.open(img_path)
        img = np.asarray(img).copy()

        # Load label (mask)
        label_path = os.path.join(self.image_dir, 'label', f'{name}.tif')
        label = Image.open(label_path)
        label = np.asarray(label).copy()

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

        # Convert to tensor format
        img = img.astype('float32') / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        label = label.astype('float32')
        label = label.transpose(2, 0, 1)  # HWC -> CHW

        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(label).float(),
            'emb_uni': self.emb_uni[idx].float(),
            'emb_conch': self.emb_conch[idx].float(),
            'emb_stpath': self.emb_stpath[idx].float(),
            'name': name
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
    """Create train, val, test dataloaders"""

    # Paths
    train_csv = os.path.join(args.hemit_dir, 'train_data.csv')
    val_csv = os.path.join(args.hemit_dir, 'val_data.csv')
    test_csv = os.path.join(args.hemit_dir, 'test_data.csv')

    train_img_dir = os.path.join(args.hemit_dir, 'he', 'train')
    val_img_dir = os.path.join(args.hemit_dir, 'he', 'val')
    test_img_dir = os.path.join(args.hemit_dir, 'he', 'test')

    # Embedding paths
    emb_uni_train = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_train.pkl')
    emb_uni_val = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_val.pkl')
    emb_uni_test = os.path.join(args.emb_dir, 'univ2emb', 'meiphi_test.pkl')

    emb_conch_train = os.path.join(args.emb_dir, 'conchemb', 'meiphi_train.pkl')
    emb_conch_val = os.path.join(args.emb_dir, 'conchemb', 'meiphi_val.pkl')
    emb_conch_test = os.path.join(args.emb_dir, 'conchemb', 'meiphi_test.pkl')

    emb_stpath_train = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_train.pkl')
    emb_stpath_val = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_val.pkl')
    emb_stpath_test = os.path.join(args.emb_dir, 'stpathinfo', 'meiphi_test.pkl')

    # Create datasets
    train_dataset = HEMITDataset(
        csv_path=train_csv,
        image_dir=train_img_dir,
        emb_uni_path=emb_uni_train,
        emb_conch_path=emb_conch_train,
        emb_stpath_path=emb_stpath_train,
        transform=get_transforms(512, is_train=True),
        stpath_feature_type=args.stpath_feature_type
    )

    val_dataset = HEMITDataset(
        csv_path=val_csv,
        image_dir=val_img_dir,
        emb_uni_path=emb_uni_val,
        emb_conch_path=emb_conch_val,
        emb_stpath_path=emb_stpath_val,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type
    )

    test_dataset = HEMITDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        emb_uni_path=emb_uni_test,
        emb_conch_path=emb_conch_test,
        emb_stpath_path=emb_stpath_test,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type
    )

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

def calculate_correlations(pred, target):
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
            p = pred[b, c].flatten()
            t = target[b, c].flatten()

            valid = ~(np.isnan(p) | np.isnan(t))
            p, t = p[valid], t[valid]

            if len(p) > 0:
                pcc, _ = pearsonr(p, t)
                scc, _ = spearmanr(p, t)
                pcc_channel.append(pcc)
                scc_channel.append(scc)

        pcc_list.append(np.nanmean(pcc_channel))
        scc_list.append(np.nanmean(scc_channel))

    return np.array(pcc_list), np.array(scc_list)


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_stpath=True):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    all_pcc = []
    all_scc = []
    expert_weights_sum = None

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        emb_uni = batch['emb_uni'].to(device)
        emb_conch = batch['emb_conch'].to(device)
        emb_stpath = batch['emb_stpath'].to(device) if use_stpath else None

        # Forward pass
        optimizer.zero_grad()
        pred, info = model(images, emb_uni, emb_conch, emb_stpath)

        # Compute loss
        loss = criterion(pred, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track expert weights
        if 'expert_weights' in info:
            w = info['expert_weights']
            if w.dim() == 1:  # Fixed weights
                w = w.detach().cpu().numpy()
            else:  # Dynamic weights [B, 3]
                w = w.mean(dim=0).detach().cpu().numpy()
            if expert_weights_sum is None:
                expert_weights_sum = w
            else:
                expert_weights_sum += w

        # Compute metrics
        pcc, scc = calculate_correlations(pred, labels)
        all_pcc.append(pcc)
        all_scc.append(scc)

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    # Average metrics
    avg_loss = total_loss / len(train_loader)
    avg_pcc = np.nanmean(all_pcc, axis=0)
    avg_scc = np.nanmean(all_scc, axis=0)

    # Average expert weights
    avg_expert_weights = None
    if expert_weights_sum is not None:
        avg_expert_weights = expert_weights_sum / len(train_loader)

    return {
        'loss': avg_loss,
        'pcc_panCK': avg_pcc[0],
        'pcc_CD3': avg_pcc[1],
        'pcc_DAPI': avg_pcc[2],
        'scc_panCK': avg_scc[0],
        'scc_CD3': avg_scc[1],
        'scc_DAPI': avg_scc[2],
        'pcc_mean': np.nanmean(avg_pcc),
        'scc_mean': np.nanmean(avg_scc),
        'expert_weights': avg_expert_weights
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, phase='Val', use_stpath=True):
    """Evaluate model"""
    model.eval()

    total_loss = 0
    all_pcc = []
    all_scc = []

    pbar = tqdm(data_loader, desc=f'[{phase}]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        emb_uni = batch['emb_uni'].to(device)
        emb_conch = batch['emb_conch'].to(device)
        emb_stpath = batch['emb_stpath'].to(device) if use_stpath else None

        # Forward pass
        pred, _ = model(images, emb_uni, emb_conch, emb_stpath)

        # Compute loss
        loss = criterion(pred, labels)

        # Compute metrics
        pcc, scc = calculate_correlations(pred, labels)
        all_pcc.append(pcc)
        all_scc.append(scc)

        total_loss += loss.item()

    # Average metrics
    avg_loss = total_loss / len(data_loader)
    avg_pcc = np.nanmean(all_pcc, axis=0)
    avg_scc = np.nanmean(all_scc, axis=0)

    return {
        'loss': avg_loss,
        'pcc_panCK': avg_pcc[0],
        'pcc_CD3': avg_pcc[1],
        'pcc_DAPI': avg_pcc[2],
        'scc_panCK': avg_scc[0],
        'scc_CD3': avg_scc[1],
        'scc_DAPI': avg_scc[2],
        'pcc_mean': np.nanmean(avg_pcc),
        'scc_mean': np.nanmean(avg_scc)
    }


def train(args):
    """Main training function"""
    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use exp_name directly (sbatch already has descriptive names)
    run_name = args.exp_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create output directory with timestamp to avoid conflicts
    exp_dir = os.path.join(args.output_dir, f"{run_name}_{timestamp}")
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

    # Get embedding dimensions (set to 0 if expert is disabled)
    if args.disable_uni:
        uni_dim = 0
        print("  UNI: DISABLED")
    else:
        uni_dim = 1536  # Fixed dim for UNI-V2
        print(f"  UNI feature dim: {uni_dim}")

    if args.disable_conch:
        conch_dim = 0
        print("  CONCH: DISABLED")
    else:
        conch_dim = 768  # Fixed dim for CONCH
        print(f"  CONCH feature dim: {conch_dim}")

    if args.disable_stpath:
        stpath_dim = 0
        print("  STPath: DISABLED")
    else:
        stpath_dim = train_loader.dataset.emb_stpath.shape[1]
        print(f"  STPath feature dim: {stpath_dim}")

    # Validate at least one expert is enabled
    num_active_experts = sum([uni_dim > 0, conch_dim > 0, stpath_dim > 0])
    if num_active_experts == 0:
        raise ValueError("At least one expert must be enabled. Cannot disable all experts.")
    print(f"  Active experts: {num_active_experts}")

    # Create model
    print("Creating model...")
    print(f"  use_cross_expert: {args.use_cross_expert}")
    print(f"  use_dynamic_gating: {args.use_dynamic_gating}")
    print(f"  use_multiscale_film: {args.use_multiscale_film}")
    print(f"  stpath_feature_type: {args.stpath_feature_type}")
    print(f"  disable_uni: {args.disable_uni}")
    print(f"  disable_conch: {args.disable_conch}")
    print(f"  disable_stpath: {args.disable_stpath}")

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
        stpath_dim=stpath_dim
    )
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    print(f"Loss type: {args.loss_type}")
    if args.loss_type in ['focal', 'weighted']:
        print(f"  Channel weights: {args.channel_weights}")
        if args.loss_type == 'focal':
            print(f"  Focal gamma: {args.focal_gamma}")
    if args.loss_type == 'mix_pearson':
        print(f"  rel_weight_channel: {args.rel_weight_channel}")
        print(f"  rel_weight_pixel: {args.rel_weight_pixel}")

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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Flag for STPath usage
    use_stpath = not args.disable_stpath

    # Training loop
    best_val_loss = float('inf')
    best_val_pcc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_stpath
        )

        # Update scheduler
        scheduler.step()

        # Log training metrics
        log_dict = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train/loss': train_metrics['loss'],
            'train/pcc_panCK': train_metrics['pcc_panCK'],
            'train/pcc_CD3': train_metrics['pcc_CD3'],
            'train/pcc_DAPI': train_metrics['pcc_DAPI'],
            'train/pcc_mean': train_metrics['pcc_mean'],
            'train/scc_panCK': train_metrics['scc_panCK'],
            'train/scc_CD3': train_metrics['scc_CD3'],
            'train/scc_DAPI': train_metrics['scc_DAPI'],
            'train/scc_mean': train_metrics['scc_mean'],
        }

        # Log expert weights (order matches enabled experts)
        if train_metrics['expert_weights'] is not None:
            w = train_metrics['expert_weights']
            enabled_experts = []
            if uni_dim > 0:
                enabled_experts.append('uni')
            if conch_dim > 0:
                enabled_experts.append('conch')
            if stpath_dim > 0:
                enabled_experts.append('stpath')
            for i, name in enumerate(enabled_experts):
                if i < len(w):
                    log_dict[f'train/expert_w_{name}'] = w[i]

        # Print training metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"PCC: {train_metrics['pcc_mean']:.4f}")
        print(f"  PCC - panCK: {train_metrics['pcc_panCK']:.4f}, "
              f"CD3: {train_metrics['pcc_CD3']:.4f}, "
              f"DAPI: {train_metrics['pcc_DAPI']:.4f}")
        if train_metrics['expert_weights'] is not None:
            w = train_metrics['expert_weights']
            # Build expert weights string based on enabled experts
            weight_parts = []
            idx = 0
            if uni_dim > 0 and idx < len(w):
                weight_parts.append(f"UNI: {w[idx]:.3f}")
                idx += 1
            if conch_dim > 0 and idx < len(w):
                weight_parts.append(f"CONCH: {w[idx]:.3f}")
                idx += 1
            if stpath_dim > 0 and idx < len(w):
                weight_parts.append(f"STPath: {w[idx]:.3f}")
            if weight_parts:
                print(f"  Expert weights - {', '.join(weight_parts)}")

        # Validate every val_every epochs
        val_metrics = None
        if epoch % args.val_every == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, 'Val', use_stpath)
            log_dict.update({
                'val/loss': val_metrics['loss'],
                'val/pcc_panCK': val_metrics['pcc_panCK'],
                'val/pcc_CD3': val_metrics['pcc_CD3'],
                'val/pcc_DAPI': val_metrics['pcc_DAPI'],
                'val/pcc_mean': val_metrics['pcc_mean'],
                'val/scc_panCK': val_metrics['scc_panCK'],
                'val/scc_CD3': val_metrics['scc_CD3'],
                'val/scc_DAPI': val_metrics['scc_DAPI'],
                'val/scc_mean': val_metrics['scc_mean'],
            })
            print(f"\nVal - Loss: {val_metrics['loss']:.4f}, "
                  f"PCC: {val_metrics['pcc_mean']:.4f}")
            print(f"  PCC - panCK: {val_metrics['pcc_panCK']:.4f}, "
                  f"CD3: {val_metrics['pcc_CD3']:.4f}, "
                  f"DAPI: {val_metrics['pcc_DAPI']:.4f}")

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)

        # Early stopping based on val loss (only when validation is run)
        if val_metrics is not None:
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_val_pcc = val_metrics['pcc_mean']
                patience_counter = 0

                # Save best model
                save_path = os.path.join(exp_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Validation metrics
                    'val_loss': val_metrics['loss'],
                    'val_pcc_panCK': val_metrics['pcc_panCK'],
                    'val_pcc_CD3': val_metrics['pcc_CD3'],
                    'val_pcc_DAPI': val_metrics['pcc_DAPI'],
                    'val_pcc_mean': val_metrics['pcc_mean'],
                    'val_scc_panCK': val_metrics['scc_panCK'],
                    'val_scc_CD3': val_metrics['scc_CD3'],
                    'val_scc_DAPI': val_metrics['scc_DAPI'],
                    'val_scc_mean': val_metrics['scc_mean'],
                    'config': vars(args)
                }, save_path)
                print(f"Saved best model (val_loss={val_metrics['loss']:.4f})")
            else:
                patience_counter += 1

        # Evaluate on test set every val_every epochs (same as val)
        if epoch % args.val_every == 0:
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test', use_stpath)
            print(f"\nTest - Loss: {test_metrics['loss']:.4f}, "
                  f"PCC: {test_metrics['pcc_mean']:.4f}")
            print(f"  PCC - panCK: {test_metrics['pcc_panCK']:.4f}, "
                  f"CD3: {test_metrics['pcc_CD3']:.4f}, "
                  f"DAPI: {test_metrics['pcc_DAPI']:.4f}")

            if not args.no_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'test/loss': test_metrics['loss'],
                    'test/pcc_panCK': test_metrics['pcc_panCK'],
                    'test/pcc_CD3': test_metrics['pcc_CD3'],
                    'test/pcc_DAPI': test_metrics['pcc_DAPI'],
                    'test/pcc_mean': test_metrics['pcc_mean'],
                    'test/scc_panCK': test_metrics['scc_panCK'],
                    'test/scc_CD3': test_metrics['scc_CD3'],
                    'test/scc_DAPI': test_metrics['scc_DAPI'],
                    'test/scc_mean': test_metrics['scc_mean'],
                })

            # Save checkpoint
            ckpt_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # Validation metrics
                'val_loss': val_metrics['loss'],
                'val_pcc_panCK': val_metrics['pcc_panCK'],
                'val_pcc_CD3': val_metrics['pcc_CD3'],
                'val_pcc_DAPI': val_metrics['pcc_DAPI'],
                'val_pcc_mean': val_metrics['pcc_mean'],
                'val_scc_panCK': val_metrics['scc_panCK'],
                'val_scc_CD3': val_metrics['scc_CD3'],
                'val_scc_DAPI': val_metrics['scc_DAPI'],
                'val_scc_mean': val_metrics['scc_mean'],
                # Test metrics
                'test_loss': test_metrics['loss'],
                'test_pcc_panCK': test_metrics['pcc_panCK'],
                'test_pcc_CD3': test_metrics['pcc_CD3'],
                'test_pcc_DAPI': test_metrics['pcc_DAPI'],
                'test_pcc_mean': test_metrics['pcc_mean'],
                'test_scc_panCK': test_metrics['scc_panCK'],
                'test_scc_CD3': test_metrics['scc_CD3'],
                'test_scc_DAPI': test_metrics['scc_DAPI'],
                'test_scc_mean': test_metrics['scc_mean'],
                'config': vars(args)
            }, ckpt_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Final test with best model
    print("\n" + "="*60)
    print("Final test with best model...")
    print("="*60)

    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', use_stpath)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  PCC Mean: {test_metrics['pcc_mean']:.4f}")
    print(f"  SCC Mean: {test_metrics['scc_mean']:.4f}")
    print(f"  PCC - panCK: {test_metrics['pcc_panCK']:.4f}, "
          f"CD3: {test_metrics['pcc_CD3']:.4f}, "
          f"DAPI: {test_metrics['pcc_DAPI']:.4f}")

    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.log({
            'final_test/loss': test_metrics['loss'],
            'final_test/pcc_panCK': test_metrics['pcc_panCK'],
            'final_test/pcc_CD3': test_metrics['pcc_CD3'],
            'final_test/pcc_DAPI': test_metrics['pcc_DAPI'],
            'final_test/pcc_mean': test_metrics['pcc_mean'],
            'final_test/scc_mean': test_metrics['scc_mean'],
        })
        wandb.finish()

    # Save test results
    import json
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_pcc': best_val_pcc,
        'test_loss': test_metrics['loss'],
        'test_pcc_panCK': test_metrics['pcc_panCK'],
        'test_pcc_CD3': test_metrics['pcc_CD3'],
        'test_pcc_DAPI': test_metrics['pcc_DAPI'],
        'test_pcc_mean': test_metrics['pcc_mean'],
        'test_scc_panCK': test_metrics['scc_panCK'],
        'test_scc_CD3': test_metrics['scc_CD3'],
        'test_scc_DAPI': test_metrics['scc_DAPI'],
        'test_scc_mean': test_metrics['scc_mean'],
        'config': {
            'use_cross_expert': args.use_cross_expert,
            'use_dynamic_gating': args.use_dynamic_gating,
            'use_multiscale_film': args.use_multiscale_film,
            'loss_type': args.loss_type,
            'focal_gamma': args.focal_gamma,
            'channel_weights': args.channel_weights,
        }
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {exp_dir}")

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
