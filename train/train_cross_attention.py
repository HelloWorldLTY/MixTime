"""
Training script for Cross Attention Fusion Model
With wandb logging and proper data loading
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
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from scipy.stats import pearsonr, spearmanr

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")

from cross_attention_model import (
    create_cross_attention_model,
    create_multi_expert_model
)


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross Attention Fusion Model')

    # Data paths
    parser.add_argument('--hemit_dir', type=str,
                        default='/work/nvme/bdxk/hzhao11/HEMIT',
                        help='Path to HEMIT dataset')
    parser.add_argument('--emb_dir', type=str,
                        default='/work/nvme/bdxk/zwang92/GigaTIME',
                        help='Path to embedding directory')
    parser.add_argument('--gigatime_weights', type=str,
                        default=None,
                        help='Path to pretrained GigaTIME weights (will download if None)')

    # Model settings
    parser.add_argument('--model_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'multi_expert'],
                        help='Model type to use')
    parser.add_argument('--freeze_gigatime', type=lambda x: x.lower() == 'true',
                        default=True, help='Freeze GigaTIME backbone (true/false)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_tokens', type=int, default=16,
                        help='Number of embedding tokens')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

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
    parser.add_argument('--exp_name', type=str, default='cross_attn_fusion',
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
        input_size=512
    ):
        self.df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.input_size = input_size

        # Load embeddings
        print(f"Loading embeddings...")
        self.emb_uni = torch.load(emb_uni_path, weights_only=False)
        self.emb_conch = torch.load(emb_conch_path, weights_only=False)
        emb_stpath_data = pd.read_pickle(emb_stpath_path)
        self.emb_stpath = emb_stpath_data['emb']

        # Concatenate embeddings
        self.emb_joint = torch.cat([
            self.emb_uni,
            self.emb_conch,
            self.emb_stpath
        ], dim=1)

        print(f"Dataset size: {len(self.df)}")
        print(f"Joint embedding shape: {self.emb_joint.shape}")

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

        # Get embeddings
        emb = self.emb_joint[idx]
        emb_uni = self.emb_uni[idx]
        emb_conch = self.emb_conch[idx]
        emb_stpath = self.emb_stpath[idx]

        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(label).float(),
            'embedding': emb.float(),
            'emb_uni': emb_uni.float(),
            'emb_conch': emb_conch.float(),
            'emb_stpath': emb_stpath.float(),
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
        transform=get_transforms(512, is_train=True)
    )

    val_dataset = HEMITDataset(
        csv_path=val_csv,
        image_dir=val_img_dir,
        emb_uni_path=emb_uni_val,
        emb_conch_path=emb_conch_val,
        emb_stpath_path=emb_stpath_val,
        transform=get_transforms(512, is_train=False)
    )

    test_dataset = HEMITDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        emb_uni_path=emb_uni_test,
        emb_conch_path=emb_conch_test,
        emb_stpath_path=emb_stpath_test,
        transform=get_transforms(512, is_train=False)
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
# Training
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, model_type='cross_attention'):
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
        if model_type == 'multi_expert':
            emb_uni = batch['emb_uni'].to(device)
            emb_conch = batch['emb_conch'].to(device)
            emb_stpath = batch['emb_stpath'].to(device)
            pred, _ = model(images, emb_uni, emb_conch, emb_stpath)
        else:
            embeddings = batch['embedding'].to(device)
            pred, _ = model(images, embeddings)

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
def evaluate(model, data_loader, criterion, device, phase='Val', model_type='cross_attention'):
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
        if model_type == 'multi_expert':
            emb_uni = batch['emb_uni'].to(device)
            emb_conch = batch['emb_conch'].to(device)
            emb_stpath = batch['emb_stpath'].to(device)
            pred, _ = model(images, emb_uni, emb_conch, emb_stpath)
        else:
            embeddings = batch['embedding'].to(device)
            pred, _ = model(images, embeddings)

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

    # Download GigaTIME weights if needed
    if args.gigatime_weights is None:
        from huggingface_hub import snapshot_download
        print("Downloading GigaTIME weights from HuggingFace...")
        repo_id = "prov-gigatime/GigaTIME"
        local_dir = snapshot_download(repo_id=repo_id)
        args.gigatime_weights = os.path.join(local_dir, "model.pth")

    # Create model
    print("Creating model...")
    if args.model_type == 'cross_attention':
        model = create_cross_attention_model(
            weights_path=args.gigatime_weights,
            emb_dim=2816,
            freeze_gigatime=args.freeze_gigatime,
            num_heads=args.num_heads,
            num_tokens=args.num_tokens,
            dropout=args.dropout
        )
    else:
        model = create_multi_expert_model(
            weights_path=args.gigatime_weights,
            freeze_gigatime=args.freeze_gigatime,
            num_heads=args.num_heads,
            num_tokens=args.num_tokens,
            dropout=args.dropout
        )

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.SmoothL1Loss()

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
            model, train_loader, criterion, optimizer, device, epoch, args.model_type
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val', args.model_type)

        # Update scheduler
        scheduler.step()

        # Log to wandb
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

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"PCC: {train_metrics['pcc_mean']:.4f}, "
              f"SCC: {train_metrics['scc_mean']:.4f}")
        print(f"  PCC - panCK: {train_metrics['pcc_panCK']:.4f}, "
              f"CD3: {train_metrics['pcc_CD3']:.4f}, "
              f"DAPI: {train_metrics['pcc_DAPI']:.4f}")

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
            test_metrics = evaluate(model, test_loader, criterion, device, 'Test', args.model_type)
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
                'val_loss': val_metrics['loss'],
                'test_loss': test_metrics['loss'],
                'test_pcc_mean': test_metrics['pcc_mean'],
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
                    'test_loss': test_metrics['loss'],
                    'test_pcc_mean': test_metrics['pcc_mean'],
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

    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', args.model_type)

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

    print(f"\nResults saved to {exp_dir}")

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
