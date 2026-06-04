"""
Channel Focal Loss for multi-channel regression tasks.

This loss automatically up-weights channels that are harder to predict,
helping prevent the model from ignoring difficult channels like CD3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelFocalLoss(nn.Module):
    """
    Pure adaptive Focal Loss for regression.

    Automatically increases weight for channels with higher loss,
    similar to how Focal Loss in classification focuses on hard examples.

    L = sum_c(focal_weight_c * loss_c)
    where focal_weight = softmax((normalized_loss)^gamma)

    This is PURELY ADAPTIVE - no fixed channel weights.
    For fixed weights, use ChannelWeightedLoss instead.

    Args:
        gamma: Focusing parameter. Higher gamma = more focus on hard channels.
               gamma=0: equal weights for all channels
               gamma=1: linear focus
               gamma=2: quadratic focus (default, same as original Focal Loss)
        base_loss: Base loss function ('smooth_l1' or 'mse')
    """

    def __init__(self, gamma=2.0, base_loss='smooth_l1'):
        super().__init__()
        self.gamma = gamma
        self.base_loss = base_loss

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]

        Returns:
            Scalar loss value
        """
        # Compute base loss per pixel: [B, C, H, W]
        if self.base_loss == 'smooth_l1':
            pixel_loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            pixel_loss = F.mse_loss(pred, target, reduction='none')

        # Average over batch and spatial dims to get per-channel loss: [C]
        channel_loss = pixel_loss.mean(dim=(0, 2, 3))

        # Compute adaptive focal weight
        # Higher loss = harder channel = higher weight
        with torch.no_grad():
            max_loss = channel_loss.max().clamp(min=1e-6)
            normalized_loss = channel_loss / max_loss  # [0, 1]
            focal_weight = normalized_loss ** self.gamma
            # Normalize weights to sum to 1
            focal_weight = focal_weight / (focal_weight.sum() + 1e-8)

        # Weighted sum of channel losses
        loss = (channel_loss * focal_weight).sum()

        return loss


class ChannelWeightedLoss(nn.Module):
    """
    Fixed weighted loss for channels.
    Use this when you want to manually specify channel weights.

    Args:
        channel_weights: Weights for each channel [panCK, CD3, DAPI]
        base_loss: Base loss function ('smooth_l1' or 'mse')
    """

    def __init__(self, channel_weights=None, base_loss='smooth_l1'):
        super().__init__()
        if channel_weights is None:
            channel_weights = [1.0, 2.0, 1.0]  # Default: higher weight for CD3
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        self.base_loss = base_loss

    def forward(self, pred, target):
        if self.base_loss == 'smooth_l1':
            pixel_loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            pixel_loss = F.mse_loss(pred, target, reduction='none')

        # Per-channel loss: [C]
        channel_loss = pixel_loss.mean(dim=(0, 2, 3))

        # Weighted sum (weights are already on correct device via register_buffer)
        loss = (channel_loss * self.channel_weights).sum() / self.channel_weights.sum()

        return loss


class MixPearsonCorrelationLoss(nn.Module):
    """
    Mix Pearson Correlation Loss adapted for image tasks.

    Balances between:
    - across channel correlation (每个像素的通道预测)
    - across pixel correlation (每个通道的空间分布)

    Original design for [genes x cells], adapted for [B, C, H, W].

    Args:
        rel_weight_channel: Weight for channel correlation loss
        rel_weight_pixel: Weight for pixel correlation loss
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        rel_weight_channel: float = 1.0,
        rel_weight_pixel: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.rel_weight_channel = rel_weight_channel
        self.rel_weight_pixel = rel_weight_pixel
        self.mse_loss = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: [B, C, H, W] predictions
            targets: [B, C, H, W] ground truth

        Returns:
            Combined loss value
        """
        B, C, H, W = preds.shape

        # Reshape to [C, B*H*W] for correlation computation
        preds_2d = preds.permute(1, 0, 2, 3).reshape(C, -1)    # [C, N] where N=B*H*W
        targets_2d = targets.permute(1, 0, 2, 3).reshape(C, -1)

        # Across channel correlation (dim=0, 跨通道)
        # 每个像素位置的3通道预测是否与目标一致
        preds_avg_channel = preds_2d.mean(dim=0, keepdim=True)
        targets_avg_channel = targets_2d.mean(dim=0, keepdim=True)
        r_channel = F.cosine_similarity(
            preds_2d - preds_avg_channel,
            targets_2d - targets_avg_channel,
            dim=0, eps=self.eps
        )  # [N]

        # Across pixel correlation (dim=1, 跨像素)
        # 每个通道的空间分布是否正确
        preds_avg_pixel = preds_2d.mean(dim=1, keepdim=True)
        targets_avg_pixel = targets_2d.mean(dim=1, keepdim=True)
        r_pixel = F.cosine_similarity(
            preds_2d - preds_avg_pixel,
            targets_2d - targets_avg_pixel,
            dim=1, eps=self.eps
        )  # [C]

        # Combined loss
        loss_channel = (1 - r_channel.mean()) * self.rel_weight_channel
        loss_pixel = (1 - r_pixel.mean()) * self.rel_weight_pixel
        loss_mse = self.mse_loss(preds, targets)

        loss = loss_channel + loss_pixel + loss_mse

        return loss


class MaskedMixPearsonLoss(nn.Module):
    """
    Mix Pearson Correlation Loss with channel mask support.

    Used for mixed dataset training where some samples only have labels for
    a subset of channels (e.g., HEMIT has 3 channels, Orion has 17).

    Supports mixed batches where different samples have different valid channels.
    Groups samples by their mask pattern and computes loss separately.

    Args:
        rel_weight_channel: Weight for channel correlation loss
        rel_weight_pixel: Weight for pixel correlation loss
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        rel_weight_channel: float = 1.0,
        rel_weight_pixel: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.rel_weight_channel = rel_weight_channel
        self.rel_weight_pixel = rel_weight_pixel

    def _compute_loss_for_channels(self, preds, targets):
        """Compute loss for given pred/target tensors (same channel mask)"""
        B, C, H, W = preds.shape

        if C == 0:
            return (preds * 0).sum()

        # Reshape to [C, B*H*W] for correlation computation
        preds_2d = preds.permute(1, 0, 2, 3).reshape(C, -1)
        targets_2d = targets.permute(1, 0, 2, 3).reshape(C, -1)

        # Across channel correlation (dim=0)
        preds_avg_channel = preds_2d.mean(dim=0, keepdim=True)
        targets_avg_channel = targets_2d.mean(dim=0, keepdim=True)
        r_channel = F.cosine_similarity(
            preds_2d - preds_avg_channel,
            targets_2d - targets_avg_channel,
            dim=0, eps=self.eps
        )

        # Across pixel correlation (dim=1)
        preds_avg_pixel = preds_2d.mean(dim=1, keepdim=True)
        targets_avg_pixel = targets_2d.mean(dim=1, keepdim=True)
        r_pixel = F.cosine_similarity(
            preds_2d - preds_avg_pixel,
            targets_2d - targets_avg_pixel,
            dim=1, eps=self.eps
        )

        # Combined loss
        loss_channel = (1 - r_channel.mean()) * self.rel_weight_channel
        loss_pixel = (1 - r_pixel.mean()) * self.rel_weight_pixel
        loss_mse = F.smooth_l1_loss(preds, targets)

        return loss_channel + loss_pixel + loss_mse

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        channel_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            preds: [B, C, H, W] predictions
            targets: [B, C, H, W] ground truth
            channel_mask: [B, C] or [C] binary mask indicating valid channels
                          If None, all channels are used.
                          Supports different masks per sample in batch.

        Returns:
            Combined loss value
        """
        B, C, H, W = preds.shape

        # Handle channel mask
        if channel_mask is None:
            # No mask, compute loss on all channels
            return self._compute_loss_for_channels(preds, targets)

        # Expand mask if needed: [C] -> [B, C]
        if channel_mask.dim() == 1:
            channel_mask = channel_mask.unsqueeze(0).expand(B, -1)

        # Check if all samples have the same mask (common case)
        first_mask = channel_mask[0]
        all_same_mask = torch.all(channel_mask == first_mask.unsqueeze(0))

        if all_same_mask:
            # Simple case: all samples have same mask
            valid_channels = first_mask.bool()
            preds_valid = preds[:, valid_channels]
            targets_valid = targets[:, valid_channels]
            return self._compute_loss_for_channels(preds_valid, targets_valid)

        # Mixed batch: group samples by mask pattern and compute loss separately
        # Convert masks to hashable tuples for grouping
        mask_to_indices = {}
        for b in range(B):
            mask_tuple = tuple(channel_mask[b].tolist())
            if mask_tuple not in mask_to_indices:
                mask_to_indices[mask_tuple] = []
            mask_to_indices[mask_tuple].append(b)

        # Compute weighted loss for each group
        total_loss = 0.0
        total_samples = 0
        for mask_tuple, indices in mask_to_indices.items():
            mask_tensor = torch.tensor(mask_tuple, dtype=torch.bool, device=preds.device)
            indices_tensor = torch.tensor(indices, device=preds.device)

            preds_group = preds[indices_tensor][:, mask_tensor]
            targets_group = targets[indices_tensor][:, mask_tensor]

            if preds_group.shape[1] > 0:  # Has valid channels
                group_loss = self._compute_loss_for_channels(preds_group, targets_group)
                total_loss = total_loss + group_loss * len(indices)
                total_samples += len(indices)

        if total_samples == 0:
            return (preds * 0).sum()

        return total_loss / total_samples


def get_loss_function(loss_type, gamma=2.0, channel_weights=None,
                      rel_weight_channel=1.0, rel_weight_pixel=1.0):
    """
    Factory function to get loss based on type.

    Args:
        loss_type: 'smooth_l1', 'focal', 'weighted', 'mix_pearson', or 'masked_mix_pearson'
        gamma: Focal loss gamma parameter (only for 'focal')
        channel_weights: Channel weights [panCK, CD3, DAPI] (only for 'weighted')
        rel_weight_channel: Weight for channel correlation (only for 'mix_pearson')
        rel_weight_pixel: Weight for pixel correlation (only for 'mix_pearson')

    Returns:
        Loss module
    """
    if loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'focal':
        # Pure adaptive - no fixed weights
        return ChannelFocalLoss(gamma=gamma)
    elif loss_type == 'weighted':
        # Fixed weights
        return ChannelWeightedLoss(channel_weights=channel_weights)
    elif loss_type == 'mix_pearson':
        return MixPearsonCorrelationLoss(
            rel_weight_channel=rel_weight_channel,
            rel_weight_pixel=rel_weight_pixel
        )
    elif loss_type == 'masked_mix_pearson':
        return MaskedMixPearsonLoss(
            rel_weight_channel=rel_weight_channel,
            rel_weight_pixel=rel_weight_pixel
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
