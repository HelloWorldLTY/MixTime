"""
Baseline Training script for GigaTIME (without embeddings)
For fair comparison with cross-attention fusion model
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

from cross_attention_model import GigaTIME, load_gigatime_pretrained
from losses.channel_focal import get_loss_function


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train GigaTIME Baseline (no embeddings)')

    # Data paths
    parser.add_argument('--hemit_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/HEMIT',
                        help='Path to HEMIT dataset')
    parser.add_argument('--gigatime_weights', type=str,
                        default=None,
                        help='Path to pretrained GigaTIME weights (will download if None)')

    # Model settings
    parser.add_argument('--model_type', type=str, default='finetune_last',
                        choices=['finetune_last', 'finetune_all'],
                        help='finetune_last: only train final layer; finetune_all: train all params')

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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output settings
    parser.add_argument('--output_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME/outputs',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default='baseline',
                        help='Experiment name')

    # Wandb settings
    parser.add_argument('--wandb_project', type=str, default='gigatime-fusion',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # Checkpoint settings
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validate every N epochs')

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
# Dataset (same as cross-attention but without embeddings)
# ============================================================================

class HEMITDatasetBaseline(Dataset):
    """HEMIT Dataset without embeddings"""

    def __init__(
        self,
        csv_path,
        image_dir,
        transform=None,
        input_size=512
    ):
        self.df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.input_size = input_size

        print(f"Dataset size: {len(self.df)}")

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

    # Create datasets
    train_dataset = HEMITDatasetBaseline(
        csv_path=train_csv,
        image_dir=train_img_dir,
        transform=get_transforms(512, is_train=True),
        input_size=512
    )

    val_dataset = HEMITDatasetBaseline(
        csv_path=val_csv,
        image_dir=val_img_dir,
        transform=get_transforms(512, is_train=False),
        input_size=512
    )

    test_dataset = HEMITDatasetBaseline(
        csv_path=test_csv,
        image_dir=test_img_dir,
        transform=get_transforms(512, is_train=False),
        input_size=512
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
# Model
# ============================================================================

class GigaTIMEBaseline(nn.Module):
    """GigaTIME baseline model for 3-channel output (panCK, CD3, DAPI)"""

    def __init__(self, gigatime_model, freeze_backbone=True):
        super().__init__()
        self.gigatime = gigatime_model

        # Replace final layer for 3 outputs
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Softplus()
        )

        # Freeze backbone if needed
        if freeze_backbone:
            for name, param in self.gigatime.named_parameters():
                if 'final' not in name:
                    param.requires_grad = False

    def forward(self, x):
        # Get features from GigaTIME (before final layer)
        feat = self.gigatime.forward_features(x)  # [B, 32, H, W]
        # Apply new final layer
        out = self.final(feat)
        return out


def create_baseline_model(weights_path=None, model_type='finetune_last'):
    """Create baseline model"""
    # Load GigaTIME with pretrained weights
    gigatime = load_gigatime_pretrained(weights_path)

    # Create baseline model
    freeze_backbone = (model_type == 'finetune_last')
    model = GigaTIMEBaseline(gigatime, freeze_backbone=freeze_backbone)

    return model


# ============================================================================
# Metrics (same as cross-attention)
# ============================================================================

def calculate_correlations(pred, target):
    """
    Calculate Pearson and Spearman correlations per channel

    Args:
        pred: [B, C, H, W] predictions
        target: [B, C, H, W] ground truth
    Returns:
        pcc: [C] Pearson correlations per channel
        scc: [C] Spearman correlations per channel
    """
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

            # Remove NaN values
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
# Training (same structure as cross-attention)
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    all_pcc = []
    all_scc = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        pred = model(images)

        # Compute loss
        loss = criterion(pred, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

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


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, phase='Val'):
    """Evaluate model"""
    model.eval()

    total_loss = 0
    all_pcc = []
    all_scc = []

    pbar = tqdm(data_loader, desc=f'[{phase}]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        pred = model(images)

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

    # Create output directory with key parameters in name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.exp_name}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}"
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

    # Get pretrained weights
    weights_path = args.gigatime_weights
    if weights_path is None:
        print("Downloading GigaTIME weights from HuggingFace...")
        from huggingface_hub import snapshot_download
        repo_id = "prov-gigatime/GigaTIME"
        local_dir = snapshot_download(repo_id=repo_id)
        weights_path = os.path.join(local_dir, "model.pth")

    # Create model
    print("Creating model...")
    model = create_baseline_model(weights_path, args.model_type)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    print(f"Loss type: {args.loss_type}")
    criterion = get_loss_function(
        loss_type=args.loss_type,
        gamma=args.focal_gamma,
        channel_weights=args.channel_weights,
        rel_weight_channel=args.rel_weight_channel,
        rel_weight_pixel=args.rel_weight_pixel
    )
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

    # Training loop
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_test_pcc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Update scheduler
        scheduler.step()

        # Log train metrics to wandb
        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log({
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
            })

        # Print train metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"PCC: {train_metrics['pcc_mean']:.4f}, "
              f"SCC: {train_metrics['scc_mean']:.4f}")
        print(f"  PCC - panCK: {train_metrics['pcc_panCK']:.4f}, "
              f"CD3: {train_metrics['pcc_CD3']:.4f}, "
              f"DAPI: {train_metrics['pcc_DAPI']:.4f}")

        # Validate every N epochs
        if epoch % args.val_every == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, 'Val')

            # Log val metrics to wandb
            if not args.no_wandb and WANDB_AVAILABLE:
                wandb.log({
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
                  f"PCC: {val_metrics['pcc_mean']:.4f}, "
                  f"SCC: {val_metrics['scc_mean']:.4f}")
            print(f"  PCC - panCK: {val_metrics['pcc_panCK']:.4f}, "
                  f"CD3: {val_metrics['pcc_CD3']:.4f}, "
                  f"DAPI: {val_metrics['pcc_DAPI']:.4f}")

            # Early stopping based on val loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

        # Save checkpoint and evaluate on test set every N epochs
        if epoch % args.save_every == 0:
            # Evaluate on test set
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test')
            print(f"\nTest - Loss: {test_metrics['loss']:.4f}, "
                  f"PCC: {test_metrics['pcc_mean']:.4f}")

            # Log test metrics to wandb
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
            print(f"Saved checkpoint to {ckpt_path}")

            # Save best model based on test performance
            if test_metrics['loss'] < best_test_loss:
                best_test_loss = test_metrics['loss']
                best_test_pcc = test_metrics['pcc_mean']
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
                }, save_path)
                print(f"Saved best model (test_loss={test_metrics['loss']:.4f}) to {save_path}")

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

    test_metrics = evaluate(model, test_loader, criterion, device, 'Test')

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  PCC Mean: {test_metrics['pcc_mean']:.4f}")
    print(f"  SCC Mean: {test_metrics['scc_mean']:.4f}")
    print(f"  PCC - panCK: {test_metrics['pcc_panCK']:.4f}, "
          f"CD3: {test_metrics['pcc_CD3']:.4f}, "
          f"DAPI: {test_metrics['pcc_DAPI']:.4f}")
    print(f"  SCC - panCK: {test_metrics['scc_panCK']:.4f}, "
          f"CD3: {test_metrics['scc_CD3']:.4f}, "
          f"DAPI: {test_metrics['scc_DAPI']:.4f}")

    # Log test results to wandb
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
        wandb.finish()

    # Save test results
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
    }

    import json
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {os.path.join(exp_dir, 'results.json')}")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
