"""
Multi-Expert Fusion V2 for GigaTIME
Configurable fusion model with modular components:
- Cross-Expert Attention: Expert interaction before combination
- Dynamic Gating: Input-dependent expert weights
- Multi-scale FiLM: Embedding injection at multiple decoder levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    """VGG-style convolutional block"""
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class GigaTIME(nn.Module):
    """U-Net++ architecture for GigaTIME"""
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, self.nb_filter[0], self.nb_filter[0])
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1])
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2])
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3])
        self.conv4_0 = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4])

        self.conv0_1 = VGGBlock(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_1 = VGGBlock(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv2_1 = VGGBlock(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])
        self.conv3_1 = VGGBlock(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3], self.nb_filter[3])

        self.conv0_2 = VGGBlock(self.nb_filter[0]*2+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_2 = VGGBlock(self.nb_filter[1]*2+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv2_2 = VGGBlock(self.nb_filter[2]*2+self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])

        self.conv0_3 = VGGBlock(self.nb_filter[0]*3+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_3 = VGGBlock(self.nb_filter[1]*3+self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])

        self.conv0_4 = VGGBlock(self.nb_filter[0]*4+self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])

        self.final = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)

    def forward_features(self, input):
        """Forward pass that returns features before final layer"""
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return x0_4  # [B, 32, H, W]

    def forward(self, input):
        features = self.forward_features(input)
        output = self.final(features)
        return output


class CrossAttnBlock(nn.Module):
    """Cross Attention block for fusing spatial features with embeddings"""

    def __init__(self, feat_dim, emb_dim, num_heads=4, num_tokens=8, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim

        # Project embedding to Key/Value tokens
        self.emb_to_kv = nn.Sequential(
            nn.Linear(emb_dim, feat_dim * num_tokens * 2),
            nn.Dropout(dropout)
        )

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer Norms
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout)
        )

    def forward(self, feat, emb):
        B, C, H, W = feat.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        q = feat.flatten(2).permute(0, 2, 1)
        q = self.norm1(q)

        # Generate K/V from embedding
        kv = self.emb_to_kv(emb)
        kv = kv.reshape(B, self.num_tokens, 2, self.feat_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        # Cross Attention
        attn_out, attn_weights = self.cross_attn(q, k, v)

        # Residual + FFN
        out = q + attn_out
        out = out + self.ffn(self.norm2(out))

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out, attn_weights


class CrossExpertInteraction(nn.Module):
    """
    Expert interaction module using self-attention.
    Allows the three experts (UNI, CONCH, STPath) to exchange information.
    """

    def __init__(self, feat_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.expert_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)

        # Generate modulation scales from attended expert features
        self.scale_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()  # Bounded output for stable modulation
        )

    def forward(self, feat_uni, feat_conch, feat_stpath):
        """
        Args:
            feat_*: [B, C, H, W] - features from each expert

        Returns:
            Modulated features for each expert
        """
        B, C, H, W = feat_uni.shape

        # Global pool to get expert representations: [B, C]
        experts = torch.stack([
            feat_uni.mean(dim=(2, 3)),
            feat_conch.mean(dim=(2, 3)),
            feat_stpath.mean(dim=(2, 3))
        ], dim=1)  # [B, 3, C]

        # Self-attention among experts
        experts_attn, _ = self.expert_attn(experts, experts, experts)
        experts_attn = self.norm(experts + experts_attn)  # [B, 3, C]

        # Generate modulation scales
        scales = self.scale_proj(experts_attn)  # [B, 3, C]
        scales = scales.unsqueeze(-1).unsqueeze(-1)  # [B, 3, C, 1, 1]

        # Modulate each expert (residual style)
        feat_uni = feat_uni * (1 + scales[:, 0])
        feat_conch = feat_conch * (1 + scales[:, 1])
        feat_stpath = feat_stpath * (1 + scales[:, 2])

        return feat_uni, feat_conch, feat_stpath


class DynamicGating(nn.Module):
    """
    Dynamic gating network that generates input-dependent expert weights.
    Uses image features to decide how to combine experts.
    """

    def __init__(self, feat_dim=32, num_experts=3, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, feat, *expert_feats):
        """
        Args:
            feat: [B, C, H, W] - original GigaTIME features (for gating)
            *expert_feats: variable number of [B, C, H, W] expert outputs

        Returns:
            feat_combined: [B, C, H, W] - weighted combination
            weights: [B, num_experts] - expert weights for this batch
        """
        # Generate dynamic weights
        logits = self.gate(feat)  # [B, num_experts]
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

        # Weighted combination
        feat_combined = sum(
            weights[:, i:i+1, None, None] * expert_feats[i]
            for i in range(len(expert_feats))
        )

        return feat_combined, weights


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Modulates features using scale (gamma) and shift (beta) from embedding.
    """

    def __init__(self, emb_dim, feat_channels):
        super().__init__()
        self.scale = nn.Linear(emb_dim, feat_channels)
        self.shift = nn.Linear(emb_dim, feat_channels)

        # Initialize to identity mapping (no change initially)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, feat, emb):
        """
        Args:
            feat: [B, C, H, W]
            emb: [B, emb_dim]

        Returns:
            Modulated features: feat * (1 + gamma) + beta
        """
        gamma = self.scale(emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.shift(emb).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]

        return feat * (1 + gamma) + beta


class MultiExpertFusionV2(nn.Module):
    """
    Unified Multi-Expert Fusion model with configurable components.

    Components (all optional):
    - Cross-Expert Attention: Expert interaction before combination
    - Dynamic Gating: Input-dependent expert weights
    - Multi-scale FiLM: Embedding injection at multiple decoder levels

    Args:
        gigatime_model: Pretrained GigaTIME model
        use_cross_expert: Enable cross-expert interaction
        use_dynamic_gating: Enable dynamic gating (else use fixed learnable weights)
        use_multiscale_film: Enable multi-scale FiLM injection
        freeze_gigatime: Freeze GigaTIME backbone
    """

    def __init__(
        self,
        gigatime_model,
        feat_dim=32,
        num_heads=4,
        num_tokens=8,
        out_channels=3,
        dropout=0.1,
        use_cross_expert=False,
        use_dynamic_gating=False,
        use_multiscale_film=False,
        freeze_gigatime=True,
        uni_dim=1536,
        conch_dim=768,
        stpath_dim=512
    ):
        super().__init__()

        self.gigatime = gigatime_model
        self.feat_dim = feat_dim
        self.use_cross_expert = use_cross_expert
        self.use_dynamic_gating = use_dynamic_gating
        self.use_multiscale_film = use_multiscale_film
        self.uni_dim = uni_dim
        self.conch_dim = conch_dim
        self.stpath_dim = stpath_dim

        # Freeze GigaTIME backbone
        if freeze_gigatime:
            for param in self.gigatime.parameters():
                param.requires_grad = False

        # Expert flags
        self.use_uni = uni_dim > 0
        self.use_conch = conch_dim > 0
        self.use_stpath = stpath_dim > 0

        # UNI expert (can be disabled with uni_dim=0)
        if self.use_uni:
            self.uni_attn = CrossAttnBlock(
                feat_dim=feat_dim, emb_dim=uni_dim, num_heads=num_heads,
                num_tokens=num_tokens, dropout=dropout
            )
        else:
            self.uni_attn = None

        # CONCH expert (can be disabled with conch_dim=0)
        if self.use_conch:
            self.conch_attn = CrossAttnBlock(
                feat_dim=feat_dim, emb_dim=conch_dim, num_heads=num_heads,
                num_tokens=num_tokens, dropout=dropout
            )
        else:
            self.conch_attn = None

        # STPath expert (can be disabled with stpath_dim=0)
        if self.use_stpath:
            # If dim is large (gene expression ~30k), add projection layer
            if stpath_dim > 1024:
                self.stpath_proj = nn.Sequential(
                    nn.Linear(stpath_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(1024, 512)
                )
                stpath_attn_dim = 512
            else:
                self.stpath_proj = None
                stpath_attn_dim = stpath_dim

            self.stpath_attn = CrossAttnBlock(
                feat_dim=feat_dim, emb_dim=stpath_attn_dim, num_heads=num_heads,
                num_tokens=num_tokens, dropout=dropout
            )
        else:
            self.stpath_proj = None
            self.stpath_attn = None

        # Count active experts
        num_experts = sum([self.use_uni, self.use_conch, self.use_stpath])
        self.num_experts = num_experts

        # Optional: Cross-Expert Interaction (only if using all 3 experts)
        if use_cross_expert and num_experts == 3:
            self.cross_expert = CrossExpertInteraction(
                feat_dim=feat_dim, num_heads=num_heads, dropout=dropout
            )
        else:
            self.cross_expert = None

        # Expert combination: Dynamic Gating or Fixed Weights
        if use_dynamic_gating:
            self.gating = DynamicGating(feat_dim=feat_dim, num_experts=num_experts)
        else:
            self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)

        # Optional: Multi-scale FiLM
        if use_multiscale_film:
            # Calculate total embedding dim based on which experts are used
            total_emb_dim = 0
            if self.use_uni:
                total_emb_dim += uni_dim
            if self.use_conch:
                total_emb_dim += conch_dim
            if self.use_stpath:
                film_stpath_dim = 512 if self.stpath_proj is not None else stpath_dim
                total_emb_dim += film_stpath_dim
            nb_filter = [32, 64, 128, 256, 512]
            self.film_layers = nn.ModuleDict({
                'x4_0': FiLMLayer(total_emb_dim, nb_filter[4]),  # bottleneck
                'x3_1': FiLMLayer(total_emb_dim, nb_filter[3]),
                'x2_2': FiLMLayer(total_emb_dim, nb_filter[2]),
                'x1_3': FiLMLayer(total_emb_dim, nb_filter[1]),
            })

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

        # Output head
        self.final = nn.Sequential(
            nn.Conv2d(feat_dim, out_channels, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x, emb_uni, emb_conch, emb_stpath=None):
        """
        Args:
            x: [B, 3, H, W] - input H&E image
            emb_uni: [B, 1536] - UNI-V2 embedding (ignored if use_uni=False)
            emb_conch: [B, 768] - CONCH embedding (ignored if use_conch=False)
            emb_stpath: [B, stpath_dim] - STPath embedding (512 or ~30k), ignored if use_stpath=False

        Returns:
            pred: [B, 3, H, W] - predictions
            info: dict with expert weights and other info
        """
        info = {}

        # Project STPath embedding if needed (for large gene expression features)
        if self.use_stpath and self.stpath_proj is not None:
            emb_stpath = self.stpath_proj(emb_stpath)

        # Multi-scale FiLM path (modify U-Net++ decoder)
        if self.use_multiscale_film:
            feat, intermediate_feats = self._forward_with_film(x, emb_uni, emb_conch, emb_stpath)
        else:
            feat = self.gigatime.forward_features(x)  # [B, 32, H, W]

        # Expert attention - collect active expert features
        expert_feats = []

        if self.use_uni:
            feat_uni, _ = self.uni_attn(feat, emb_uni)
            expert_feats.append(feat_uni)

        if self.use_conch:
            feat_conch, _ = self.conch_attn(feat, emb_conch)
            expert_feats.append(feat_conch)

        if self.use_stpath:
            feat_stpath, _ = self.stpath_attn(feat, emb_stpath)
            expert_feats.append(feat_stpath)

        # Optional: Cross-expert interaction (only with all 3 experts)
        if self.cross_expert is not None and self.num_experts == 3:
            feat_uni, feat_conch, feat_stpath = self.cross_expert(
                expert_feats[0], expert_feats[1], expert_feats[2]
            )
            expert_feats = [feat_uni, feat_conch, feat_stpath]

        # Expert combination
        if self.use_dynamic_gating:
            feat_attended, weights = self.gating(feat, *expert_feats)
            info['expert_weights'] = weights  # [B, num_experts]
        else:
            w = torch.softmax(self.expert_weights, dim=0)
            feat_attended = sum(w[i] * expert_feats[i] for i in range(len(expert_feats)))
            info['expert_weights'] = w  # [num_experts]

        # Fuse with original features
        feat_fused = self.fusion(torch.cat([feat, feat_attended], dim=1))

        # Output
        pred = self.final(feat_fused)

        return pred, info

    def _forward_with_film(self, x, emb_uni, emb_conch, emb_stpath):
        """
        Forward pass with multi-scale FiLM injection.
        Manually unrolls U-Net++ to inject FiLM at decoder stages.
        Note: emb_stpath should already be projected if needed (done in forward())
        """
        g = self.gigatime

        # Concatenate embeddings for FiLM (only include enabled experts)
        emb_list = []
        if self.use_uni:
            emb_list.append(emb_uni)
        if self.use_conch:
            emb_list.append(emb_conch)
        if self.use_stpath and emb_stpath is not None:
            emb_list.append(emb_stpath)
        emb_concat = torch.cat(emb_list, dim=1)

        # Encoder
        x0_0 = g.conv0_0(x)
        x1_0 = g.conv1_0(g.pool(x0_0))
        x2_0 = g.conv2_0(g.pool(x1_0))
        x3_0 = g.conv3_0(g.pool(x2_0))
        x4_0 = g.conv4_0(g.pool(x3_0))

        # FiLM at bottleneck
        x4_0 = self.film_layers['x4_0'](x4_0, emb_concat)

        # Decoder level 3
        x0_1 = g.conv0_1(torch.cat([x0_0, g.up(x1_0)], 1))
        x3_1 = g.conv3_1(torch.cat([x3_0, g.up(x4_0)], 1))
        x3_1 = self.film_layers['x3_1'](x3_1, emb_concat)

        # Decoder level 2
        x1_1 = g.conv1_1(torch.cat([x1_0, g.up(x2_0)], 1))
        x2_1 = g.conv2_1(torch.cat([x2_0, g.up(x3_0)], 1))
        x2_2 = g.conv2_2(torch.cat([x2_0, x2_1, g.up(x3_1)], 1))
        x2_2 = self.film_layers['x2_2'](x2_2, emb_concat)

        # Decoder level 1
        x0_2 = g.conv0_2(torch.cat([x0_0, x0_1, g.up(x1_1)], 1))
        x1_2 = g.conv1_2(torch.cat([x1_0, x1_1, g.up(x2_1)], 1))
        x1_3 = g.conv1_3(torch.cat([x1_0, x1_1, x1_2, g.up(x2_2)], 1))
        x1_3 = self.film_layers['x1_3'](x1_3, emb_concat)

        # Final decoder level
        x0_3 = g.conv0_3(torch.cat([x0_0, x0_1, x0_2, g.up(x1_2)], 1))
        x0_4 = g.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, g.up(x1_3)], 1))

        return x0_4, None

    def get_config(self):
        """Return configuration dict for logging"""
        return {
            'use_cross_expert': self.use_cross_expert,
            'use_dynamic_gating': self.use_dynamic_gating,
            'use_multiscale_film': self.use_multiscale_film,
        }


def load_gigatime_pretrained(weights_path=None):
    """Load GigaTIME model with pretrained weights"""
    model = GigaTIME(num_classes=23, input_channels=3)

    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights_path}")

    return model


def create_multi_expert_v2(
    weights_path=None,
    freeze_gigatime=True,
    num_heads=4,
    num_tokens=8,
    dropout=0.1,
    use_cross_expert=False,
    use_dynamic_gating=False,
    use_multiscale_film=False,
    uni_dim=1536,
    conch_dim=768,
    stpath_dim=512,
    out_channels=3
):
    """Factory function to create MultiExpertFusionV2 model

    Args:
        uni_dim: UNI-V2 embedding dimension (0 to disable)
        conch_dim: CONCH embedding dimension (0 to disable)
        stpath_dim: STPath embedding dimension (512 for emb, ~30k for gene expression, 0 to disable)
        out_channels: Number of output channels (3 for HEMIT, 17 for Orion)
    """
    gigatime = load_gigatime_pretrained(weights_path)

    model = MultiExpertFusionV2(
        gigatime_model=gigatime,
        feat_dim=32,
        num_heads=num_heads,
        num_tokens=num_tokens,
        out_channels=out_channels,
        dropout=dropout,
        use_cross_expert=use_cross_expert,
        use_dynamic_gating=use_dynamic_gating,
        use_multiscale_film=use_multiscale_film,
        freeze_gigatime=freeze_gigatime,
        uni_dim=uni_dim,
        conch_dim=conch_dim,
        stpath_dim=stpath_dim
    )

    return model


if __name__ == "__main__":
    # Test the model with different configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        {'use_cross_expert': False, 'use_dynamic_gating': False, 'use_multiscale_film': False},
        {'use_cross_expert': True, 'use_dynamic_gating': False, 'use_multiscale_film': False},
        {'use_cross_expert': False, 'use_dynamic_gating': True, 'use_multiscale_film': False},
        {'use_cross_expert': False, 'use_dynamic_gating': False, 'use_multiscale_film': True},
        {'use_cross_expert': True, 'use_dynamic_gating': True, 'use_multiscale_film': True},
    ]

    for cfg in configs:
        print(f"\nConfig: {cfg}")
        model = create_multi_expert_v2(
            weights_path=None,
            freeze_gigatime=True,
            **cfg
        )
        model = model.to(device)

        # Test input
        x = torch.randn(2, 3, 256, 256).to(device)
        emb_uni = torch.randn(2, 1536).to(device)
        emb_conch = torch.randn(2, 768).to(device)
        emb_stpath = torch.randn(2, 512).to(device)

        # Forward pass
        pred, info = model(x, emb_uni, emb_conch, emb_stpath)

        print(f"  Output shape: {pred.shape}")
        print(f"  Expert weights: {info['expert_weights']}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")
