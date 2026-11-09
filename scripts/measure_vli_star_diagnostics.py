#!/usr/bin/env python3
"""Measure oracle VLI* diagnostics on trained checkpoints.

This script uses private client training labels to measure epsilon_train, r,
and Q. It is an analysis/verification script, not an attacker-side estimator.
Run it from VLI_in_FL so the repository's data cache paths resolve correctly.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataSampling import LocalDataset, get_dataloaders_Dirichlet
from models import get_model
from client.attacks import VLI_star_closed_form, VLI_star_generalized


DATASET_CLASSES = {"CIFAR10": 10, "CIFAR100": 100, "SVHN": 10, "MNIST": 10, "Tiny": 200}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more model state_dict checkpoint paths")
    parser.add_argument("--dataset", choices=DATASET_CLASSES, default="CIFAR10")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=5,
                        help="Number of client batches in one analyzed window")
    parser.add_argument("--prop", type=float, default=0.1,
                        help="Stratified fraction of each test class used as aux")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--client-ids", default="0",
                        help="Comma-separated client IDs or 'all'")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", default="results/vli_star_diagnostics")
    return parser.parse_args()


def get_labels(dataset):
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("dataset must expose targets or labels")
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    return np.asarray(labels, dtype=np.int64)


def make_datasets(dataset_name, batch_size, n_clients, alpha, seed, prop, n_classes):
    train_loaders, _ = get_dataloaders_Dirichlet(
        n_clients=n_clients,
        alpha=alpha,
        rand_seed=seed,
        dataset=dataset_name,
        batch_size=batch_size,
    )
    data_dir = "./data/"
    if dataset_name == "CIFAR10":
        full_test = datasets.CIFAR10(data_dir, train=False, download=True,
                                     transform=train_loaders[0].dataset.dataset.transform)
    elif dataset_name == "CIFAR100":
        full_test = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=train_loaders[0].dataset.dataset.transform)
    elif dataset_name == "SVHN":
        full_test = datasets.SVHN(data_dir, split="test", download=True,
                                  transform=train_loaders[0].dataset.dataset.transform)
    elif dataset_name == "MNIST":
        full_test = datasets.MNIST(data_dir, train=False, download=True,
                                   transform=train_loaders[0].dataset.dataset.transform)
    elif dataset_name == "Tiny":
        from torchvision.datasets import ImageFolder
        full_test = ImageFolder("./data/tiny-imagenet-200/val",
                                transform=train_loaders[0].dataset.dataset.transform)

    labels = get_labels(full_test)
    rng = np.random.default_rng(seed)
    selected = []
    for cls in range(n_classes):
        ids = np.flatnonzero(labels == cls)
        count = min(len(ids), max(1, int(prop * len(ids))))
        if count:
            selected.extend(rng.choice(ids, size=count, replace=False).tolist())
    aux_dataset = LocalDataset(full_test, selected)
    return train_loaders, aux_dataset


@torch.inference_mode()
def measure(model, dataset, n_classes, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_count = np.zeros(n_classes, dtype=np.int64)
    class_error_mass = np.zeros(n_classes, dtype=np.float64)
    class_probability_mass = np.zeros((n_classes, n_classes), dtype=np.float64)
    loss_sum = 0.0
    correct = 0
    total = 0

    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        logits, _ = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        errors = 1.0 - true_probability
        loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())

        labels_np = labels.cpu().numpy()
        errors_np = errors.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        for cls in range(n_classes):
            mask = labels_np == cls
            if np.any(mask):
                class_count[cls] += int(mask.sum())
                class_error_mass[cls] += float(errors_np[mask].sum())
                class_probability_mass[cls] += probabilities_np[mask].sum(axis=0)

    if total == 0:
        raise ValueError("cannot measure an empty dataset")
    epsilon = float(class_error_mass.sum() / total)
    class_epsilon = np.full(n_classes, np.nan, dtype=np.float64)
    present = class_count > 0
    class_epsilon[present] = class_error_mass[present] / class_count[present]
    r = np.full(n_classes, np.nan, dtype=np.float64)
    if epsilon > 0:
        r[present] = class_epsilon[present] / epsilon

    Q = np.full((n_classes, n_classes), np.nan, dtype=np.float64)
    for cls in np.flatnonzero(present):
        row = class_probability_mass[cls].copy()
        row[cls] = 0.0
        error_mass = class_error_mass[cls]
        if error_mass > 0:
            Q[cls] = row / error_mass
    kappa = float(np.max(r[present]) / np.min(r[present])) if np.any(present) and np.min(r[present]) > 0 else float("inf")
    bias_gradient = class_probability_mass.sum(axis=0) - class_count
    gradient_l1 = float(np.abs(bias_gradient).sum())
    gradient_bound_ratio = gradient_l1 / (2.0 * total * epsilon) if epsilon > 0 else float("nan")
    return {
        "epsilon": epsilon,
        "cross_entropy": loss_sum / total,
        "error_rate": 1.0 - correct / total,
        "count": total,
        "class_count": class_count,
        "class_epsilon": class_epsilon,
        "r": r,
        "Q": Q,
        "kappa": kappa,
        "gradient_l1": gradient_l1,
        "gradient_bound_ratio": gradient_bound_ratio,
        "epsilon_to_ce_ratio": epsilon / (loss_sum / total) if loss_sum > 0 else float("nan"),
    }


@torch.inference_mode()
def collect_predictions(model, dataset, batch_size, device):
    labels_all, probabilities_all = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    for inputs, labels in loader:
        logits, _ = model(inputs.to(device))
        probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy().astype(np.int64))
    if not labels_all:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float64)
    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def class_error_geometry(labels, probabilities, n_classes, fallback_Q=None):
    errors = 1.0 - probabilities[np.arange(len(labels)), labels]
    counts = np.bincount(labels, minlength=n_classes)
    error_mass = np.bincount(labels, weights=errors, minlength=n_classes)
    epsilon = float(error_mass.sum() / max(len(labels), 1))
    r = np.ones(n_classes, dtype=np.float64)
    Q = np.zeros((n_classes, n_classes), dtype=np.float64)
    present = counts > 0
    if epsilon > 0:
        r[present] = (error_mass[present] / counts[present]) / epsilon
    for cls in range(n_classes):
        if error_mass[cls] > 1e-15:
            row = probabilities[labels == cls].sum(axis=0).copy()
            row[cls] = 0.0
            Q[cls] = row / error_mass[cls]
        elif fallback_Q is not None:
            Q[cls] = fallback_Q[cls]
        else:
            Q[cls] = 1.0 / (n_classes - 1)
            Q[cls, cls] = 0.0
    return counts, epsilon, r, Q


def evaluate_windows(model, dataset, aux_dataset, n_classes, batch_size,
                     window_size, device, checkpoint, client):
    labels, probabilities = collect_predictions(model, dataset, batch_size, device)
    aux_labels, aux_probabilities = collect_predictions(model, aux_dataset, batch_size, device)
    _, _, _, aux_Q = class_error_geometry(aux_labels, aux_probabilities, n_classes)
    records = []
    for index in range(len(labels) // window_size):
        start = index * window_size
        stop = start + window_size
        y = labels[start:stop]
        p = probabilities[start:stop]
        truth, epsilon, r, Q = class_error_geometry(y, p, n_classes, fallback_Q=aux_Q)
        signal = p.sum(axis=0) - np.bincount(y, minlength=n_classes)
        row = {
            "checkpoint": checkpoint,
            "client": client,
            "window": index,
            "n_samples": window_size,
            "present_classes": int(np.count_nonzero(truth)),
            "epsilon_window": epsilon,
            "closed_iAcc": "",
            "closed_cAcc": "",
            "generalized_iAcc_oracle": "",
            "generalized_cAcc_oracle": "",
            "generalized_relative_residual": "",
            "uniform_r_iAcc_with_oracle_Q": "",
            "uniform_r_theory_lower_bound": "",
            "status": "ok",
        }
        try:
            closed = VLI_star_closed_form(signal, n_samples=window_size)
            row["closed_iAcc"] = float(np.minimum(closed, truth).sum() / window_size)
            row["closed_cAcc"] = float(np.mean((closed > 0) == (truth > 0)))
        except (ValueError, FloatingPointError) as exc:
            row["status"] = f"closed_skipped: {exc}"
        try:
            generalized, diagnostics = VLI_star_generalized(
                signal, r, Q, n_samples=window_size, return_diagnostics=True,
            )
            row["generalized_iAcc_oracle"] = float(np.minimum(generalized, truth).sum() / window_size)
            row["generalized_cAcc_oracle"] = float(np.mean((generalized > 0) == (truth > 0)))
            row["generalized_relative_residual"] = diagnostics["relative_residual"]
            present = truth > 0
            kappa = (
                float(np.max(r[present]) / np.min(r[present]))
                if np.any(present) and np.min(r[present]) > 0 else float("inf")
            )
            uniform_r = np.ones(n_classes, dtype=np.float64)
            uniform_prediction = VLI_star_generalized(signal, uniform_r, Q, n_samples=window_size)
            row["uniform_r_iAcc_with_oracle_Q"] = float(np.minimum(uniform_prediction, truth).sum() / window_size)
            row["uniform_r_theory_lower_bound"] = 1.0 / kappa if np.isfinite(kappa) and kappa > 0 else 0.0
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row["status"] = f"{row['status']}; generalized_skipped: {exc}"
        records.append(row)
    return records


def load_checkpoint(path, model, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state_dict: {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def main():
    args = parse_args()
    n_classes = args.n_classes or DATASET_CLASSES[args.dataset]
    if n_classes != DATASET_CLASSES[args.dataset]:
        raise ValueError(f"{args.dataset} requires n_classes={DATASET_CLASSES[args.dataset]}")
    if not 0 < args.prop <= 1:
        raise ValueError("prop must be in (0, 1]")
    if args.local_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("batch-size and local-epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    train_loaders, aux_dataset = make_datasets(
        args.dataset, args.batch_size, args.n_clients, args.alpha, args.seed,
        args.prop, n_classes,
    )
    if args.client_ids.strip().lower() == "all":
        client_ids = list(range(args.n_clients))
    else:
        client_ids = [int(value) for value in args.client_ids.split(",")]
    if any(client < 0 or client >= args.n_clients for client in client_ids):
        raise ValueError("client ID outside [0, n_clients)")

    channel = 1 if args.dataset == "MNIST" else 3
    model = get_model(
        model_name=args.model,
        net_params=(n_classes, channel, args.hidden),
        device=device,
        n_hidden=1,
        n_dim=300,
        batchnorm=False,
        dropout=True,
        tanh=(args.activation == "tanh"),
        leaky_relu=False,
    ).to(device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    q_path = output_dir / "Q_rows.csv"
    windows_path = output_dir / "window_recovery.csv"
    summary_fields = [
        "checkpoint", "client", "epsilon_train", "train_cross_entropy", "train_error_rate",
        "epsilon_aux", "aux_cross_entropy", "aux_error_rate", "kappa_train",
        "train_bias_gradient_l1", "train_gradient_bound_ratio",
        "train_epsilon_to_ce_ratio", "aux_heldout_gap_slack",
        "present_classes", "absent_classes", "aux_examples", "device",
        "diagnostic_status",
    ]

    window_fields = [
        "checkpoint", "client", "window", "n_samples", "present_classes",
        "epsilon_window", "closed_iAcc", "closed_cAcc",
        "generalized_iAcc_oracle", "generalized_cAcc_oracle",
        "generalized_relative_residual", "uniform_r_iAcc_with_oracle_Q",
        "uniform_r_theory_lower_bound", "status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file, \
            q_path.open("w", newline="", encoding="utf-8") as q_file, \
            windows_path.open("w", newline="", encoding="utf-8") as windows_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        q_writer = csv.writer(q_file)
        windows_writer = csv.DictWriter(windows_file, fieldnames=window_fields)
        summary_writer.writeheader()
        q_writer.writerow(["checkpoint", "client", "source_class", "target_class", "Q_jk", "row_source"])
        windows_writer.writeheader()

        for checkpoint in args.checkpoint:
            checkpoint_path = str(Path(checkpoint).resolve())
            load_checkpoint(checkpoint_path, model, device)
            aux = measure(model, aux_dataset, n_classes, args.batch_size, device)

            for client in client_ids:
                local_dataset = train_loaders[client].dataset
                train = measure(model, local_dataset, n_classes, args.batch_size, device)
                window_rows = evaluate_windows(
                    model, local_dataset, aux_dataset, n_classes, args.batch_size,
                    args.batch_size * args.local_epochs, device, checkpoint_path, client,
                )
                windows_writer.writerows(window_rows)
                present = np.flatnonzero(train["class_count"] > 0)
                absent = np.flatnonzero(train["class_count"] == 0)
                summary_writer.writerow({
                    "checkpoint": checkpoint_path,
                    "client": client,
                    "epsilon_train": f"{train['epsilon']:.10g}",
                    "train_cross_entropy": f"{train['cross_entropy']:.10g}",
                    "train_error_rate": f"{train['error_rate']:.10g}",
                    "epsilon_aux": f"{aux['epsilon']:.10g}",
                    "aux_cross_entropy": f"{aux['cross_entropy']:.10g}",
                    "aux_error_rate": f"{aux['error_rate']:.10g}",
                    "kappa_train": f"{train['kappa']:.10g}",
                    "train_bias_gradient_l1": f"{train['gradient_l1']:.10g}",
                    "train_gradient_bound_ratio": f"{train['gradient_bound_ratio']:.10g}",
                    "train_epsilon_to_ce_ratio": f"{train['epsilon_to_ce_ratio']:.10g}",
                    "aux_heldout_gap_slack": f"{aux['epsilon'] - aux['error_rate'] / 2:.10g}",
                    "present_classes": ";".join(map(str, present.tolist())),
                    "absent_classes": ";".join(map(str, absent.tolist())),
                    "aux_examples": aux["count"],
                    "device": str(device),
                    "diagnostic_status": "oracle: uses private client labels",
                })

                # Q rows for classes with nonzero train error come from private
                # train data. Rows with zero error mass do not affect Q^T a; use
                # class-conditional aux rows for completeness and mark their source.
                aux_labels = get_labels(aux_dataset.dataset)
                aux_ids_by_class = {
                    cls: set(np.flatnonzero(aux_labels == cls).tolist())
                    for cls in range(n_classes)
                }
                for cls in range(n_classes):
                    row_source = "train"
                    row = train["Q"][cls]
                    if cls not in present or not np.all(np.isfinite(row)):
                        row_source = "aux_class_conditional_absent_or_zero_error"
                        class_ids = aux_ids_by_class[cls]
                        mapped_ids = [i for i, global_id in enumerate(aux_dataset.idxs) if global_id in class_ids]
                        class_aux = LocalDataset(aux_dataset, mapped_ids)
                        row = measure(model, class_aux, n_classes, args.batch_size, device)["Q"][cls]
                    for target in range(n_classes):
                        q_writer.writerow([
                            checkpoint_path, client, cls, target,
                            "" if not np.isfinite(row[target]) else f"{row[target]:.10g}",
                            row_source,
                        ])

    print(f"Wrote summary: {summary_path.resolve()}")
    print(f"Wrote Q rows: {q_path.resolve()}")
    print(f"Wrote window recovery: {windows_path.resolve()}")
    print("Train-derived r/Q and window scores are oracle diagnostics; they do not show that r or Q are attacker-estimable.")


if __name__ == "__main__":
    main()
