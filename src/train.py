"""
train.py — Training pipeline for the Self-Pruning Neural Network.

Usage
-----
# Run full λ sweep defined in config.yaml:
    python train.py

# Override a single λ value:
    python train.py --lambda_sparse 1e-4

# Quick smoke-test (5 epochs):
    python train.py --epochs 5 --lambda_sparse 1e-4

# Custom config:
    python train.py --config my_config.yaml
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

# Allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))

from model import SelfPruningMLP
from evaluate import evaluate_model, print_sparsity_report
from utils import (
    AverageMeter,
    get_device,
    load_checkpoint,
    plot_accuracy_vs_sparsity,
    plot_gate_distributions,
    save_checkpoint,
    set_seed,
)

# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data                                                                        #
# --------------------------------------------------------------------------- #

def get_cifar10_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
#  Training loop                                                               #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: SelfPruningMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lambda_sparse: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    One epoch of training.

    Returns (total_loss, ce_loss, sparsity_loss_raw, train_accuracy%).

    Total Loss = CrossEntropyLoss + λ * SparsityLoss
    """
    model.train()
    total_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    sp_loss_meter = AverageMeter()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        ce_loss = criterion(logits, targets)

        # SparsityLoss = sum of all gate values (L1 on positive sigmoid outputs)
        sp_loss = model.sparsity_loss()
        loss = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        # Clip gradients to prevent instability with large λ values
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_meter.update(loss.item(), batch_size)
        ce_loss_meter.update(ce_loss.item(), batch_size)
        sp_loss_meter.update(sp_loss.item(), batch_size)

        _, predicted = logits.max(dim=1)
        total += batch_size
        correct += predicted.eq(targets).sum().item()

    train_acc = 100.0 * correct / total
    return total_loss_meter.avg, ce_loss_meter.avg, sp_loss_meter.avg, train_acc


# --------------------------------------------------------------------------- #
#  Single-lambda experiment                                                    #
# --------------------------------------------------------------------------- #

def train_model(
    config: dict,
    lambda_sparse: float,
    device: torch.device,
) -> tuple[float, float, float, object]:
    """
    Train one model for a given λ.

    Returns (final_test_acc, final_sparsity, best_test_acc, all_gate_values).
    """
    run_name = f"lambda_{lambda_sparse:.0e}"
    set_seed(config["seed"])

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Training run: {run_name}")
    logger.info(f"  λ = {lambda_sparse:.1e}  |  epochs = {config['epochs']}")
    logger.info("=" * 60)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"],
        data_dir=config["data_dir"],
    )

    model = SelfPruningMLP(
        input_dim=3072,
        hidden_dims=config["hidden_dims"],
        num_classes=10,
        dropout=config.get("dropout", 0.3),
    ).to(device)

    param_info = model.param_count()
    logger.info(
        f"  Params — total: {param_info['total']:,}  |  "
        f"weights: {param_info['weight_params']:,}  |  "
        f"gate scores: {param_info['gate_params']:,}"
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    # Checkpoint directory for this run
    ckpt_dir = Path(config["checkpoint_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    log_every = max(1, config["epochs"] // 10)  # Log ~10 times per run

    for epoch in range(1, config["epochs"] + 1):
        train_loss, ce_loss, sp_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_sparse, device
        )
        test_acc, sparsity = evaluate_model(model, test_loader, device)
        scheduler.step()

        if epoch % log_every == 0 or epoch == config["epochs"] or epoch == 1:
            logger.info(
                f"  [{epoch:3d}/{config['epochs']}] "
                f"Loss: {train_loss:.4f} (CE: {ce_loss:.4f} | SP: {sp_loss:.1f}) | "
                f"Train: {train_acc:.1f}% | Test: {test_acc:.2f}% | "
                f"Sparsity: {sparsity * 100:.1f}%"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                model, optimizer, epoch, test_acc, sparsity,
                ckpt_dir / "best.pt",
            )

    # Final evaluation
    final_acc, final_sparsity = evaluate_model(model, test_loader, device)
    all_gates = model.get_all_gates().cpu().numpy()

    logger.info("")
    logger.info(f"  ✓ Final Test Accuracy : {final_acc:.2f}%")
    logger.info(f"  ✓ Final Sparsity      : {final_sparsity * 100:.2f}%")
    logger.info(f"  ✓ Best Test Accuracy  : {best_acc:.2f}%")

    print_sparsity_report(model)

    return final_acc, final_sparsity, best_acc, all_gates


# --------------------------------------------------------------------------- #
#  Multi-lambda sweep                                                          #
# --------------------------------------------------------------------------- #

def run_experiments(config: dict) -> list[dict]:
    """
    Run training for every λ in config['lambda_values'].
    Saves results.csv and both plots to experiments/.
    """
    device = get_device()
    logger.info(f"Device: {device}")

    results_dir = Path(config["results_dir"])
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    results_csv = results_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Test Accuracy (%)", "Sparsity (%)", "Best Accuracy (%)"])

    all_results = []
    all_gates_dict = {}

    for lam in config["lambda_values"]:
        final_acc, final_sparsity, best_acc, gates = train_model(config, lam, device)

        row = {
            "lambda": lam,
            "accuracy": final_acc,
            "sparsity": final_sparsity * 100,
            "best_accuracy": best_acc,
        }
        all_results.append(row)
        all_gates_dict[lam] = gates

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{lam:.1e}",
                f"{final_acc:.2f}",
                f"{final_sparsity * 100:.2f}",
                f"{best_acc:.2f}",
            ])

    # ------------------------------------------------------------------ #
    #  Summary table                                                       #
    # ------------------------------------------------------------------ #
    logger.info("")
    logger.info("=" * 60)
    logger.info("  EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Lambda':<12} {'Test Acc':>10} {'Sparsity%':>12} {'Best Acc':>10}")
    logger.info(f"  {'-'*46}")
    for r in all_results:
        logger.info(
            f"  {r['lambda']:<12.1e} {r['accuracy']:>10.2f} "
            f"{r['sparsity']:>12.2f} {r['best_accuracy']:>10.2f}"
        )
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    #  Plots                                                               #
    # ------------------------------------------------------------------ #
    logger.info("")
    logger.info("Generating visualizations...")
    plot_gate_distributions(all_gates_dict, plots_dir)
    plot_accuracy_vs_sparsity(all_results, plots_dir)

    logger.info(f"\nResults → {results_csv}")
    logger.info(f"Plots   → {plots_dir}")

    return all_results


# --------------------------------------------------------------------------- #
#  CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--lambda_sparse", type=float, default=None,
                        help="Run a single λ value (overrides config sweep)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.lambda_sparse is not None:
        config["lambda_values"] = [args.lambda_sparse]
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed

    logger.info("Config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    run_experiments(config)


if __name__ == "__main__":
    main()
