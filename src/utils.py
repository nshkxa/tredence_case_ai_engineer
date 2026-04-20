"""
utils.py — Shared utilities: reproducibility, device selection,
           checkpointing, metrics, and matplotlib visualizations.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """Pin all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
#  Device selection                                                            #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU — training will be slow")
    return device


# --------------------------------------------------------------------------- #
#  Checkpoint helpers                                                          #
# --------------------------------------------------------------------------- #

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    sparsity: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
            "sparsity": sparsity,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: Path,
    device: torch.device,
) -> tuple[int, float, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["accuracy"], ckpt["sparsity"]


# --------------------------------------------------------------------------- #
#  Running average meter                                                       #
# --------------------------------------------------------------------------- #

class AverageMeter:
    """Tracks running mean of a scalar (loss, accuracy, etc.)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# --------------------------------------------------------------------------- #
#  Visualizations                                                              #
# --------------------------------------------------------------------------- #

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]


def plot_gate_distributions(gates_dict: dict, save_dir: Path) -> None:
    """
    One histogram per lambda value showing the final gate distribution.
    A successful result shows a spike near 0 (pruned) and a cluster
    of values away from 0 (active weights).
    """
    save_dir = Path(save_dir)
    lambdas = list(gates_dict.keys())
    n = len(lambdas)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, lam, color in zip(axes, lambdas, PALETTE):
        gates = np.array(gates_dict[lam])
        near_zero_pct = float(np.mean(gates < 1e-2) * 100)

        ax.hist(gates, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(x=0.01, color="crimson", linestyle="--", linewidth=1.2, label="Threshold (0.01)")

        ax.set_title(f"λ = {lam:.0e}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        ax.text(
            0.97, 0.95,
            f"Pruned: {near_zero_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
        )
        ax.legend(fontsize=9)

    fig.suptitle(
        "Gate Value Distributions — Self-Pruning Network",
        fontsize=15, fontweight="bold", y=1.03,
    )
    plt.tight_layout()

    out_path = save_dir / "gate_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_accuracy_vs_sparsity(results: list[dict], save_dir: Path) -> None:
    """
    Two subplots:
      Left  — scatter of Test Accuracy vs Sparsity, annotated by λ
      Right — grouped bar chart of Accuracy & Sparsity per λ
    """
    save_dir = Path(save_dir)
    lambdas = [r["lambda"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Accuracy vs Sparsity scatter ---
    sc = ax1.scatter(
        sparsities, accuracies,
        c=range(len(lambdas)), cmap="RdYlGn_r",
        s=220, zorder=5, edgecolors="black", linewidths=1.4,
    )
    ax1.plot(sparsities, accuracies, "--", color="gray", alpha=0.45, zorder=4)

    for s, a, lam in zip(sparsities, accuracies, lambdas):
        ax1.annotate(
            f"λ={lam:.0e}", (s, a),
            textcoords="offset points", xytext=(10, 6), fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax1.set_xlabel("Sparsity (%)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Sparsity Trade-off", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")

    # --- Right: Grouped bar chart ---
    x = np.arange(len(lambdas))
    w = 0.35

    bars1 = ax2.bar(x - w / 2, accuracies, w, label="Test Accuracy (%)", color="#2196F3", alpha=0.85, edgecolor="black")
    bars2 = ax2.bar(x + w / 2, sparsities, w, label="Sparsity (%)", color="#FF5722", alpha=0.85, edgecolor="black")

    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}", ha="center", va="bottom", fontsize=9,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l:.0e}" for l in lambdas])
    ax2.set_xlabel("Lambda (λ)", fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.set_title("Accuracy & Sparsity per Lambda", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Self-Pruning Network — Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = save_dir / "accuracy_vs_sparsity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
