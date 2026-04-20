"""
evaluate.py — Evaluation helpers: accuracy, sparsity, per-layer stats.
"""

import torch
import numpy as np

from model import PrunableLinear


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 1e-2,
) -> tuple[float, float]:
    """
    Returns (test_accuracy %, overall_sparsity fraction).

    Sparsity = fraction of weights where gate < threshold.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    sparsity = model.overall_sparsity(threshold=threshold)
    return accuracy, sparsity


def get_gate_statistics(
    model: torch.nn.Module, threshold: float = 1e-2
) -> dict:
    """
    Per-layer gate statistics: mean, std, min, max, sparsity%.
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates().cpu().numpy().flatten()
            stats[name] = {
                "mean": float(np.mean(gates)),
                "std": float(np.std(gates)),
                "min": float(np.min(gates)),
                "max": float(np.max(gates)),
                "sparsity_pct": float(np.mean(gates < threshold) * 100),
                "total_weights": int(len(gates)),
                "pruned_weights": int(np.sum(gates < threshold)),
            }
    return stats


def print_sparsity_report(model: torch.nn.Module, threshold: float = 1e-2) -> None:
    """Pretty-print per-layer and overall sparsity."""
    stats = get_gate_statistics(model, threshold)
    header = f"{'Layer':<35} {'Total':>8} {'Pruned':>8} {'Sparsity%':>10} {'Mean Gate':>10}"
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("SPARSITY REPORT")
    print(sep)
    print(header)
    print(sep)
    for name, s in stats.items():
        print(
            f"{name:<35} {s['total_weights']:>8} {s['pruned_weights']:>8} "
            f"{s['sparsity_pct']:>10.1f} {s['mean']:>10.4f}"
        )
    print(sep)
    overall = model.overall_sparsity(threshold) * 100
    print(f"{'Overall Sparsity':.<35} {overall:.2f}%")
    print(f"{sep}\n")
