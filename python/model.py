"""GeoNet: SE-ResNet dual-head network for geometry theorem proving.

Architecture:
  - Input: 20×32×32 tensor (from state_to_tensor())
  - Backbone: SE-ResNet with 6 residual blocks, 128 channels
  - Value head: (v_logit, k) for V = tanh(v_logit + k * delta_D)
  - Policy head: 512 logits over construction index space

~3M parameters.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensor dimensions from encoding.rs
NUM_CHANNELS = 20
GRID_SIZE = 32

# Construction types that are actually generated (src/construction.rs)
CONSTRUCTION_TYPES = [
    "Midpoint",             # 0
    "Altitude",             # 1
    "Circumcenter",         # 2
    "Orthocenter",          # 3
    "Incenter",             # 4
    "ParallelThrough",      # 5
    "PerpendicularThrough", # 6
]
NUM_CONSTRUCTION_TYPES = len(CONSTRUCTION_TYPES)

# Policy head outputs 2048 logits. Each construction type gets 292 slots.
# 7 types × 292 = 2044 used slots (+4 unused = 2048 total).
# Larger space reduces hash collisions for problems with many objects.
SLOTS_PER_TYPE = 292
POLICY_SIZE = 2048


def construction_to_index(type_name: str, args: list[int]) -> int:
    """Map a construction (type, args) to a policy index in [0, 512).

    For 2-arg types (Midpoint): sort both args (order doesn't matter).
    For 3-arg types: preserve first arg (the point), sort last two (the line).
    This avoids collisions like Altitude(0,1,2) vs Altitude(1,0,2).
    """
    type_id = CONSTRUCTION_TYPES.index(type_name)
    if len(args) == 2:
        canonical = tuple(sorted(args))
    elif len(args) >= 3:
        # First arg is the main point, remaining define the line
        canonical = (args[0], *sorted(args[1:]))
    else:
        canonical = tuple(args)
    h = 0
    for a in canonical:
        h = h * 37 + a + 1
    return type_id * SLOTS_PER_TYPE + (h % SLOTS_PER_TYPE)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        s = x.mean(dim=[2, 3])  # (B, C)
        # Excitation
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, C)
        return x * s.view(b, c, 1, 1)


class SEResBlock(nn.Module):
    """Residual block with squeeze-excite attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class GeoNet(nn.Module):
    """SE-ResNet dual-head network for geometry theorem proving.

    Input: (B, 20, 32, 32) tensor from state_to_tensor()
    Output:
      - v_logit: (B,) raw value before tanh
      - k: (B,) confidence scalar for delta_D weighting
      - policy_logits: (B, 512) logits over construction space
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        backbone_channels: int = 128,
        num_blocks: int = 6,
        se_reduction: int = 8,
        policy_size: int = POLICY_SIZE,
    ):
        super().__init__()

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, backbone_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(backbone_channels)

        # SE-ResNet backbone
        self.blocks = nn.Sequential(
            *[SEResBlock(backbone_channels, se_reduction) for _ in range(num_blocks)]
        )

        # Value head: global avg pool → FC → (v_logit, k)
        self.value_conv = nn.Conv2d(backbone_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 2)  # (v_logit, k)

        # Policy head: conv → FC → logits
        self.policy_conv = nn.Conv2d(backbone_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(2 * GRID_SIZE * GRID_SIZE, 256)
        self.policy_fc2 = nn.Linear(256, policy_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize value head output to small values
        nn.init.xavier_uniform_(self.value_fc2.weight, gain=0.01)
        # Initialize k bias to 0.5 (reasonable default for delta_D weighting)
        with torch.no_grad():
            self.value_fc2.bias[1] = 0.5

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 20, 32, 32) state tensor
            valid_mask: (B, 512) boolean mask of valid construction indices.
                        If None, all logits are returned unmasked.

        Returns:
            v_logit: (B,) raw value
            k: (B,) delta_D confidence scalar
            policy_logits: (B, 512) construction logits (masked if valid_mask given)
        """
        # Backbone
        h = F.relu(self.input_bn(self.input_conv(x)))
        h = self.blocks(h)  # (B, 128, 32, 32)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h)))  # (B, 1, 32, 32)
        v = v.view(v.size(0), -1)  # (B, 1024)
        v = F.relu(self.value_fc1(v))  # (B, 128)
        v_out = self.value_fc2(v)  # (B, 2)
        v_logit = v_out[:, 0]
        k = v_out[:, 1]

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))  # (B, 2, 32, 32)
        p = p.view(p.size(0), -1)  # (B, 2048)
        p = F.relu(self.policy_fc1(p))  # (B, 256)
        policy_logits = self.policy_fc2(p)  # (B, 512)

        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_mask, float("-inf"))

        return v_logit, k, policy_logits

    def predict(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float, torch.Tensor]:
        """Single-state prediction for inference (no grad, returns numpy-friendly)."""
        self.eval()
        with torch.no_grad():
            v_logit, k, logits = self.forward(x.unsqueeze(0), valid_mask.unsqueeze(0) if valid_mask is not None else None)
            policy = F.softmax(logits[0], dim=0)
            return v_logit.item(), k.item(), policy

    def compute_value(self, v_logit: float, k: float, delta_d: float) -> float:
        """Compute final value: V = tanh(v_logit + k * delta_D)."""
        return math.tanh(v_logit + k * delta_d)


def tensor_from_flat(flat: list[float], device: str = "cpu") -> torch.Tensor:
    """Convert flat 20480-element list from encode_state() to (20, 32, 32) tensor."""
    return torch.tensor(flat, dtype=torch.float32, device=device).view(
        NUM_CHANNELS, GRID_SIZE, GRID_SIZE
    )


def build_valid_mask(constructions, device: str = "cpu") -> torch.Tensor:
    """Build a boolean mask of valid policy indices from a list of PyConstruction objects."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for c in constructions:
        idx = construction_to_index(c.construction_type(), c.args())
        mask[idx] = True
    return mask


def constructions_to_policy_target(
    constructions,
    visit_counts: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """Convert MCTS visit counts to a policy target distribution.

    Args:
        constructions: list of PyConstruction from generate_constructions()
        visit_counts: visit count for each construction (same order)
        device: torch device

    Returns:
        (512,) probability distribution over policy indices
    """
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(visit_counts)
    if total == 0:
        return target
    for c, v in zip(constructions, visit_counts):
        idx = construction_to_index(c.construction_type(), c.args())
        target[idx] += v / total
    return target


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
