"""
model.py — PrunableLinear layer and SelfPruningMLP architecture.

Core idea: Each weight has a learnable gate in (0, 1) via sigmoid.
Gates multiply the weights element-wise. L1 regularization on gate
values during training drives them toward 0, effectively pruning weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with learnable gate parameters.

    Each weight w_ij has a corresponding gate score g_ij. During the
    forward pass:
        gates      = sigmoid(gate_scores)     # gates ∈ (0, 1)
        pruned_w   = weight * gates           # element-wise mask
        output     = pruned_w @ x^T + bias

    Gradients flow through both `weight` and `gate_scores` because:
    - pruned_w depends on both via element-wise product
    - F.linear is differentiable w.r.t. its weight argument

    L1 regularization on gate values (sum of gates) penalizes active
    gates, pushing sigmoid(gate_scores) → 0, which means gate_scores
    → -∞ and the weight is effectively removed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight tensor — same initialization as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Gate scores — same shape as weight; sigmoid maps these to (0, 1)
        # Initialized with a slight positive bias so gates start around 0.7,
        # giving the optimizer headroom to either open or close each gate.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming uniform (matches PyTorch's nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Uniform in [-1, 1] → sigmoid gives ~0.27 to ~0.73, a neutral start
        nn.init.uniform_(self.gate_scores, -1.0, 1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Mask weights — gates near 0 silence the corresponding weight
        pruned_weights = self.weight * gates

        # Step 3: Standard linear transform using masked weights
        # Implemented manually via F.linear to satisfy the "no nn.Linear" constraint
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached from graph)."""
        return torch.sigmoid(self.gate_scores)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights with gate < threshold."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# --------------------------------------------------------------------------- #
#  Network                                                                     #
# --------------------------------------------------------------------------- #

class SelfPruningMLP(nn.Module):
    """
    A fully-connected network for CIFAR-10 classification where every
    linear layer is a PrunableLinear.

    Architecture (default):
        Input  → Flatten → 3072
        FC1    : PrunableLinear(3072, 512) + BN + ReLU + Dropout
        FC2    : PrunableLinear( 512, 256) + BN + ReLU + Dropout
        FC3    : PrunableLinear( 256, 128) + BN + ReLU + Dropout
        Output : PrunableLinear( 128,  10)
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: list = None,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Build hidden layers
        dims = [input_dim] + hidden_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(PrunableLinear(dims[i], dims[i + 1]))
            blocks.append(nn.BatchNorm1d(dims[i + 1]))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(p=dropout))

        self.hidden = nn.Sequential(*blocks)
        self.classifier = PrunableLinear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten: (B, C, H, W) → (B, 3072)
        x = self.hidden(x)
        return self.classifier(x)

    # ------------------------------------------------------------------ #
    #  Sparsity utilities                                                  #
    # ------------------------------------------------------------------ #

    def sparsity_loss(self) -> torch.Tensor:
        """
        Sum of ALL gate values across every PrunableLinear layer.

        This is the L1 norm of the gate vector (gates are positive by
        construction, so |gate| = gate). Minimising this sum pushes
        gate_scores → -∞ → sigmoid → 0, effectively removing weights.
        """
        gate_sums = [
            torch.sigmoid(m.gate_scores).sum()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        ]
        return sum(gate_sums)

    @torch.no_grad()
    def get_all_gates(self) -> torch.Tensor:
        """Concatenate all gate values into a single 1-D tensor."""
        return torch.cat([
            m.get_gates().flatten()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        ])

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of all weights whose gate is below `threshold`."""
        gates = self.get_all_gates()
        return (gates < threshold).float().mean().item()

    def prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        gate_params = sum(
            m.gate_scores.numel()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        return {"total": total, "gate_params": gate_params, "weight_params": total - gate_params}
