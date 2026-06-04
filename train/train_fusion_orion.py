"""
Training script for Multi-Expert Fusion V2 Model on Orion Dataset
Supports 17-channel IF prediction from H&E images using UNI/CONCH/STPath embeddings
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
    raise ImportError("tifffile is required for Orion dataset. Install with: pip install tifffile")

from models.multi_expert_v2 import create_multi_expert_v2
from losses.channel_focal import get_loss_function, ChannelFocalLoss


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


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Expert Fusion V2 Model on Orion Dataset')

    # Data paths
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
                        help='Directory with preprocessed .npy labels (faster loading)')

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
    parser.add_argument('--exp_name', type=str, default='orion_fusion_17ch',
                        help='Experiment name')

    # Wandb settings
    parser.add_argument('--wandb_project', type=str, default='gigatime-orion',
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
# Dataset
# ============================================================================

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
    """Create train, val, test dataloaders for Orion dataset"""

    # Paths
    train_csv = os.path.join(args.orion_dir, 'train_dataframe.csv')
    val_csv = os.path.join(args.orion_dir, 'val_dataframe.csv')
    test_csv = os.path.join(args.orion_dir, 'test_dataframe.csv')

    # Embedding paths (orion_*.pkl)
    emb_uni_train = os.path.join(args.emb_dir, 'univ2emb', 'orion_train.pkl')
    emb_uni_val = os.path.join(args.emb_dir, 'univ2emb', 'orion_val.pkl')
    emb_uni_test = os.path.join(args.emb_dir, 'univ2emb', 'orion_test.pkl')

    emb_conch_train = os.path.join(args.emb_dir, 'conchemb', 'orion_train.pkl')
    emb_conch_val = os.path.join(args.emb_dir, 'conchemb', 'orion_val.pkl')
    emb_conch_test = os.path.join(args.emb_dir, 'conchemb', 'orion_test.pkl')

    emb_stpath_train = os.path.join(args.emb_dir, 'stpathinfo', 'orion_train.pkl')
    emb_stpath_val = os.path.join(args.emb_dir, 'stpathinfo', 'orion_val.pkl')
    emb_stpath_test = os.path.join(args.emb_dir, 'stpathinfo', 'orion_test.pkl')

    # Create datasets
    train_dataset = OrionDataset(
        csv_path=train_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_train,
        emb_conch_path=emb_conch_train,
        emb_stpath_path=emb_stpath_train,
        transform=get_transforms(512, is_train=True),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
    )

    val_dataset = OrionDataset(
        csv_path=val_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_val,
        emb_conch_path=emb_conch_val,
        emb_stpath_path=emb_stpath_val,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
    )

    test_dataset = OrionDataset(
        csv_path=test_csv,
        data_dir=args.orion_dir,
        emb_uni_path=emb_uni_test,
        emb_conch_path=emb_conch_test,
        emb_stpath_path=emb_stpath_test,
        transform=get_transforms(512, is_train=False),
        stpath_feature_type=args.stpath_feature_type,
        npy_dir=args.npy_dir
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

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            pred, info = model(images, emb_uni, emb_conch, emb_stpath)
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
            if w.dim() == 1:  # Fixed weights
                w = w.detach().cpu().numpy()
            else:  # Dynamic weights [B, 3]
                w = w.mean(dim=0).detach().cpu().numpy()
            if expert_weights_sum is None:
                expert_weights_sum = w
            else:
                expert_weights_sum += w

        # Skip correlation computation during training (expensive CPU operation)
        # PCC/SCC will be computed during validation only

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    # Average metrics
    avg_loss = total_loss / len(train_loader)
    # Use placeholder values for training metrics (actual PCC/SCC computed in validation)
    avg_pcc = np.zeros(len(ORION_CHANNEL_NAMES))
    avg_scc = np.zeros(len(ORION_CHANNEL_NAMES))

    # Average expert weights
    avg_expert_weights = None
    if expert_weights_sum is not None:
        avg_expert_weights = expert_weights_sum / len(train_loader)

    # Format metrics
    metrics = format_channel_metrics(avg_pcc, avg_scc, ORION_CHANNEL_NAMES)
    metrics['loss'] = avg_loss
    metrics['expert_weights'] = avg_expert_weights

    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, phase='Val', use_stpath=True, use_amp=True, compute_metrics=True):
    """Evaluate model with optional AMP support

    Args:
        compute_metrics: If False, only compute loss (fast). If True, also compute PCC/SCC (slow).
    """
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

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            pred, _ = model(images, emb_uni, emb_conch, emb_stpath)
            loss = criterion(pred, labels)

        # Compute metrics only if requested (expensive CPU operation)
        if compute_metrics:
            pcc, scc = calculate_correlations(pred, labels)
            all_pcc.append(pcc)
            all_scc.append(scc)

        total_loss += loss.item()

    # Average metrics
    avg_loss = total_loss / len(data_loader)

    if compute_metrics:
        avg_pcc = np.nanmean(all_pcc, axis=0)
        avg_scc = np.nanmean(all_scc, axis=0)
        metrics = format_channel_metrics(avg_pcc, avg_scc, ORION_CHANNEL_NAMES)
    else:
        # Return placeholder metrics when not computing
        metrics = format_channel_metrics(
            np.zeros(len(ORION_CHANNEL_NAMES)),
            np.zeros(len(ORION_CHANNEL_NAMES)),
            ORION_CHANNEL_NAMES
        )

    metrics['loss'] = avg_loss
    return metrics


def train(args):
    """Main training function"""
    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use exp_name directly
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

    # Create model with 17 output channels
    print("Creating model...")
    print(f"  out_channels: {OUT_CHANNELS}")
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
        stpath_dim=stpath_dim,
        out_channels=OUT_CHANNELS  # 17 channels for Orion
    )
    model = model.to(device)

    # Multi-GPU support with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

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

    # AMP GradScaler for mixed precision training
    scaler = GradScaler()
    use_amp = True
    print(f"AMP (Automatic Mixed Precision): Enabled")

    # Training loop
    best_val_loss = float('inf')
    best_val_pcc = 0
    patience_counter = 0
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # Load model state
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load AMP scaler state
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Resume from next epoch
        start_epoch = checkpoint['epoch'] + 1

        # Restore best metrics and patience counter
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        if 'best_val_pcc' in checkpoint:
            best_val_pcc = checkpoint['best_val_pcc']
        elif 'val_pcc_mean' in checkpoint:
            best_val_pcc = checkpoint['val_pcc_mean']
        if 'patience_counter' in checkpoint:
            patience_counter = checkpoint['patience_counter']

        # Get current lr from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
        print(f"  Current LR: {current_lr:.2e}")
        print(f"  Best val_loss: {best_val_loss:.4f}, patience_counter: {patience_counter}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train with AMP
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_stpath, scaler
        )

        # Update scheduler
        scheduler.step()

        # Log training metrics
        log_dict = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train/loss': train_metrics['loss'],
            'train/pcc_mean': train_metrics['pcc_mean'],
            'train/scc_mean': train_metrics['scc_mean'],
        }

        # Log per-channel metrics
        for name in ORION_CHANNEL_NAMES:
            log_dict[f'train/pcc_{name}'] = train_metrics[f'pcc_{name}']
            log_dict[f'train/scc_{name}'] = train_metrics[f'scc_{name}']

        # Log expert weights (dynamic based on enabled experts)
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
        print(f"  Top-3 PCC: {ORION_CHANNEL_NAMES[0]}: {train_metrics[f'pcc_{ORION_CHANNEL_NAMES[0]}']:.4f}, "
              f"{ORION_CHANNEL_NAMES[1]}: {train_metrics[f'pcc_{ORION_CHANNEL_NAMES[1]}']:.4f}, "
              f"{ORION_CHANNEL_NAMES[2]}: {train_metrics[f'pcc_{ORION_CHANNEL_NAMES[2]}']:.4f}")
        if train_metrics['expert_weights'] is not None:
            w = train_metrics['expert_weights']
            enabled_experts = []
            if uni_dim > 0:
                enabled_experts.append('UNI')
            if conch_dim > 0:
                enabled_experts.append('CONCH')
            if stpath_dim > 0:
                enabled_experts.append('STPath')
            weight_str = ", ".join([f"{name}: {w[i]:.3f}" for i, name in enumerate(enabled_experts) if i < len(w)])
            print(f"  Expert weights - {weight_str}")

        # Validate every epoch (loss only) for best model selection
        # Full metrics (PCC/SCC) computed every val_every epochs
        compute_full_metrics = (epoch % args.val_every == 0)
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val', use_stpath, use_amp,
                               compute_metrics=compute_full_metrics)
        log_dict['val/loss'] = val_metrics['loss']

        if compute_full_metrics:
            log_dict['val/pcc_mean'] = val_metrics['pcc_mean']
            log_dict['val/scc_mean'] = val_metrics['scc_mean']
            for name in ORION_CHANNEL_NAMES:
                log_dict[f'val/pcc_{name}'] = val_metrics[f'pcc_{name}']
                log_dict[f'val/scc_{name}'] = val_metrics[f'scc_{name}']
            print(f"\nVal - Loss: {val_metrics['loss']:.4f}, PCC: {val_metrics['pcc_mean']:.4f}")
            print(f"  Top-3 PCC: {ORION_CHANNEL_NAMES[0]}: {val_metrics[f'pcc_{ORION_CHANNEL_NAMES[0]}']:.4f}, "
                  f"{ORION_CHANNEL_NAMES[1]}: {val_metrics[f'pcc_{ORION_CHANNEL_NAMES[1]}']:.4f}, "
                  f"{ORION_CHANNEL_NAMES[2]}: {val_metrics[f'pcc_{ORION_CHANNEL_NAMES[2]}']:.4f}")
        else:
            print(f"\nVal - Loss: {val_metrics['loss']:.4f}")

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict)

        # Early stopping based on val loss (every epoch now)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if compute_full_metrics:
                best_val_pcc = val_metrics['pcc_mean']
            patience_counter = 0

            # Save best model (handle DataParallel wrapper)
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
                save_dict['val_scc_mean'] = val_metrics['scc_mean']
                for name in ORION_CHANNEL_NAMES:
                    save_dict[f'val_pcc_{name}'] = val_metrics[f'pcc_{name}']
                    save_dict[f'val_scc_{name}'] = val_metrics[f'scc_{name}']
            torch.save(save_dict, save_path)
            print(f"Saved best model (val_loss={val_metrics['loss']:.4f})")
        else:
            patience_counter += 1

        # Save checkpoint every epoch (for resume capability)
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
        # Add validation metrics (loss always, PCC/SCC only on full metric epochs)
        ckpt_dict['val_loss'] = val_metrics['loss']
        if compute_full_metrics:
            ckpt_dict['val_pcc_mean'] = val_metrics['pcc_mean']
            ckpt_dict['val_scc_mean'] = val_metrics['scc_mean']
            for name in ORION_CHANNEL_NAMES:
                ckpt_dict[f'val_pcc_{name}'] = val_metrics[f'pcc_{name}']
                ckpt_dict[f'val_scc_{name}'] = val_metrics[f'scc_{name}']
        torch.save(ckpt_dict, ckpt_path)

        # Delete old checkpoint to save space (keep only last 2)
        old_ckpt = os.path.join(exp_dir, f'checkpoint_epoch_{epoch - 2}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Final test with best model
    print("\n" + "="*60)
    print("Final test with best model...")
    print("="*60)

    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device, weights_only=False)
    # Handle DataParallel wrapper when loading
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, 'Test', use_stpath, use_amp)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  PCC Mean: {test_metrics['pcc_mean']:.4f}")
    print(f"  SCC Mean: {test_metrics['scc_mean']:.4f}")
    print(f"\nPer-channel PCC:")
    for i, name in enumerate(ORION_CHANNEL_NAMES):
        print(f"  {name}: {test_metrics[f'pcc_{name}']:.4f}")

    if not args.no_wandb and WANDB_AVAILABLE:
        final_log = {
            'final_test/loss': test_metrics['loss'],
            'final_test/pcc_mean': test_metrics['pcc_mean'],
            'final_test/scc_mean': test_metrics['scc_mean'],
        }
        for name in ORION_CHANNEL_NAMES:
            final_log[f'final_test/pcc_{name}'] = test_metrics[f'pcc_{name}']
            final_log[f'final_test/scc_{name}'] = test_metrics[f'scc_{name}']
        wandb.log(final_log)
        wandb.finish()

    # Save test results
    import json
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_pcc': best_val_pcc,
        'test_loss': test_metrics['loss'],
        'test_pcc_mean': test_metrics['pcc_mean'],
        'test_scc_mean': test_metrics['scc_mean'],
        'config': {
            'use_cross_expert': args.use_cross_expert,
            'use_dynamic_gating': args.use_dynamic_gating,
            'use_multiscale_film': args.use_multiscale_film,
            'loss_type': args.loss_type,
            'focal_gamma': args.focal_gamma,
            'channel_weights': args.channel_weights,
            'out_channels': OUT_CHANNELS,
        }
    }
    # Add per-channel test results
    for name in ORION_CHANNEL_NAMES:
        results[f'test_pcc_{name}'] = test_metrics[f'pcc_{name}']
        results[f'test_scc_{name}'] = test_metrics[f'scc_{name}']

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {exp_dir}")

    return results


if __name__ == '__main__':
    args = parse_args()
    train(args)
