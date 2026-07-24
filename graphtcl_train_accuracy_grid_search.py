
import argparse
import gc
import itertools
import random
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_loader import load_data
from model import *
from modules import *

warnings.filterwarnings("ignore", category=UserWarning)
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)
class GraphWithTopoDataset(Dataset):
    """Dataset returning a graph, its topological feature vector, and its label."""

    def __init__(
        self,
        graphs: Sequence,
        topo_feats: torch.Tensor,
        labels: torch.Tensor,
    ):
        if not (len(graphs) == len(topo_feats) == len(labels)):
            raise ValueError(
                "graphs, topo_feats, and labels must have the same length."
            )
        self.graphs = graphs
        self.topo_feats = topo_feats
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.topo_feats[idx], self.labels[idx]


class EncoderWithDropout(nn.Module):
    """
    Adds output dropout to an existing encoder.

    This makes --dropout_list effective without requiring changes to
    GINEncoder or MLPEncoder in model.py/modules.py.
    """

    def __init__(self, encoder: nn.Module, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        # Preserve commonly used metadata when it exists.
        if hasattr(encoder, "out_dim"):
            self.out_dim = encoder.out_dim

    def forward(self, *args, **kwargs):
        embedding = self.encoder(*args, **kwargs)
        return self.dropout(embedding)


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric NT-Xent-style loss between graph and topology embeddings."""
    if z1.size(0) != z2.size(0):
        raise ValueError("z1 and z2 must have the same batch size.")

    z1 = F.normalize(z1, p=2, dim=-1, eps=eps)
    z2 = F.normalize(z2, p=2, dim=-1, eps=eps)

    batch_size = z1.size(0)
    embeddings = torch.cat([z1, z2], dim=0)

    sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature

    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=sim_matrix.device,
    )
    sim_matrix = sim_matrix.masked_fill(
        self_mask,
        torch.finfo(sim_matrix.dtype).min,
    )

    positive_indices = torch.cat(
        [
            torch.arange(
                batch_size,
                2 * batch_size,
                device=sim_matrix.device,
            ),
            torch.arange(
                0,
                batch_size,
                device=sim_matrix.device,
            ),
        ]
    )

    return F.cross_entropy(sim_matrix, positive_indices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a reproducible NumPy/Python seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_grad_scaler(use_amp: bool):
    """Create a GradScaler using the current API, with an older-version fallback."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def make_loader(
    graphs: Sequence,
    topo_features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    use_cuda: bool,
    seed: int,
) -> DataLoader:
    graph_dataset = GraphWithTopoDataset(
        graphs=graphs,
        topo_feats=topo_features,
        labels=labels,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "dataset": graph_dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def build_model(
    node_feat_dim: int,
    topo_dim: int,
    hidden_channels: int,
    dropout: float,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Build one model for one fold.

    The constructor signatures follow the GINEncoder, MLPEncoder, and
    HybridGraphTopoModel usage in the supplied code.
    """
    graph_encoder = GINEncoder(
        in_dim=node_feat_dim,
        hidden_dim=hidden_channels,
        out_dim=hidden_channels,
    )

    topo_encoder = MLPEncoder(
        in_dim=topo_dim,
        hidden_dim=hidden_channels,
        out_dim=hidden_channels,
    )

    # Apply the searched dropout value to both encoder outputs.
    graph_encoder = EncoderWithDropout(graph_encoder, dropout)
    topo_encoder = EncoderWithDropout(topo_encoder, dropout)

    model = HybridGraphTopoModel(
        graph_encoder=graph_encoder,
        topo_encoder=topo_encoder,
        hidden_dim=hidden_channels,
        proj_dim=hidden_channels,
        out_dim=num_classes,
    )

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
    alpha: float,
    temperature: float,
    max_grad_norm: float,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    The returned accuracy is the online mini-batch accuracy. Early stopping
    below uses a separate full training-set evaluation after the epoch.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_graphs, batch_topo, batch_labels in loader:
        batch_graphs = batch_graphs.to(device, non_blocking=(device.type == "cuda"))
        batch_topo = batch_topo.to(
            device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        )
        batch_labels = batch_labels.to(
            device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        ).view(-1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, _, _ = model(
                batch_graphs.x,
                batch_graphs.edge_index,
                batch_graphs.batch,
                batch_topo,
            )

            classification_loss = F.cross_entropy(
                logits,
                batch_labels,
            )

            # Kept consistent with the original model usage.
            graph_embedding = model.graph_encoder(
                batch_graphs.x,
                batch_graphs.edge_index,
                batch_graphs.batch,
            )
            topo_embedding = model.topo_encoder(batch_topo)

            alignment_loss = contrastive_loss(
                graph_embedding,
                topo_embedding,
                temperature=temperature,
            )

            loss = classification_loss + alpha * alignment_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = batch_graphs.num_graphs
        total_loss += loss.detach().item() * batch_size
        total_correct += (
            logits.detach().argmax(dim=-1) == batch_labels
        ).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()

    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch_graphs, batch_topo, batch_labels in loader:
            batch_graphs = batch_graphs.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            batch_topo = batch_topo.to(
                device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            batch_labels = batch_labels.to(
                device,
                dtype=torch.long,
                non_blocking=(device.type == "cuda"),
            ).view(-1)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, _, _ = model(
                    batch_graphs.x,
                    batch_graphs.edge_index,
                    batch_graphs.batch,
                    batch_topo,
                )

            predictions = logits.argmax(dim=-1)
            total_correct += (
                predictions == batch_labels
            ).sum().item()
            total_examples += batch_graphs.num_graphs

    if total_examples == 0:
        return 0.0

    return total_correct / total_examples

def run_hyperparameter_combination(
    args,
    dataset,
    topo_tensor,
    labels,
    cv_splits,
    hidden_channels,
    lr,
    alpha,
    node_feat_dim,
    topo_dim,
    num_classes,
    device,
):
    """
    Run all folds for one hyperparameter combination.

    Early stopping is based only on full training-set accuracy. Whenever
    training accuracy improves, the current test accuracy is recorded.
    That recorded test accuracy becomes the fold accuracy.
    """
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and not args.no_amp

    fold_accuracies: List[float] = []
    fold_best_train_accuracies: List[float] = []
    fold_best_epochs: List[int] = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv_splits,
        start=1,
    ):
        fold_seed = args.seed + fold_number
        set_seed(fold_seed)

        train_graphs = [dataset[int(i)] for i in train_idx]
        test_graphs = [dataset[int(i)] for i in test_idx]

        topo_train_raw = topo_tensor[train_idx]
        topo_test_raw = topo_tensor[test_idx]

        # Fit normalization on the training fold only.
        train_mean = topo_train_raw.mean(dim=0, keepdim=True)
        train_std = topo_train_raw.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        )

        topo_train = (
            topo_train_raw - train_mean
        ) / (train_std + 1e-6)

        topo_test = (
            topo_test_raw - train_mean
        ) / (train_std + 1e-6)

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        train_loader = make_loader(
            graphs=train_graphs,
            topo_features=topo_train,
            labels=y_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            use_cuda=use_cuda,
            seed=fold_seed,
        )

        # This loader measures the final model's accuracy on the full
        # training fold after each epoch.
        train_eval_loader = make_loader(
            graphs=train_graphs,
            topo_features=topo_train,
            labels=y_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_cuda=use_cuda,
            seed=fold_seed,
        )

        test_loader = make_loader(
            graphs=test_graphs,
            topo_features=topo_test,
            labels=y_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_cuda=use_cuda,
            seed=fold_seed,
        )

        model = build_model(
            node_feat_dim=node_feat_dim,
            topo_dim=topo_dim,
            hidden_channels=hidden_channels,
            dropout=args.dropout,
            num_classes=num_classes,
            device=device,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
        )

        scaler = make_grad_scaler(use_amp)

        best_train_acc = -float("inf")
        corresponding_test_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, online_train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                alpha=alpha,
                temperature=args.temperature,
                max_grad_norm=args.max_grad_norm,
                use_amp=use_amp,
            )

            # Important: use evaluation mode on the full training fold.
            current_train_acc = evaluate_accuracy(
                model=model,
                loader=train_eval_loader,
                device=device,
                use_amp=use_amp,
            )

            improved = (
                current_train_acc
                > best_train_acc + args.min_delta
            )

            if improved:
                best_train_acc = current_train_acc
                best_epoch = epoch
                epochs_without_improvement = 0

                # Test accuracy from exactly the epoch selected by
                # training accuracy.
                corresponding_test_acc = evaluate_accuracy(
                    model=model,
                    loader=test_loader,
                    device=device,
                    use_amp=use_amp,
                )
            else:
                epochs_without_improvement += 1

            #if args.verbose or improved or epoch == 1:
                # print(
                #     f"    Fold {fold_number:02d} | "
                #     f"Epoch {epoch:03d} | "
                #     f"Loss {train_loss:.4f} | "
                #     f"Online train {online_train_acc:.4f} | "
                #     f"Full train {current_train_acc:.4f} | "
                #     f"Best train {best_train_acc:.4f} | "
                #     f"Test@best-train "
                #     f"{corresponding_test_acc:.4f} | "
                #     f"Patience "
                #     f"{epochs_without_improvement}/"
                #     f"{args.patience}"
                # )

            if epochs_without_improvement >= args.patience:
                print(
                    f"    Early stopping at epoch {epoch}; "
                    f"best training accuracy was reached at "
                    f"epoch {best_epoch}."
                )
                break

        fold_accuracies.append(float(corresponding_test_acc))
        fold_best_train_accuracies.append(float(best_train_acc))
        fold_best_epochs.append(int(best_epoch))

        print(
            f"  Fold {fold_number:02d}/{len(cv_splits)} | "
            f"Best train accuracy: {best_train_acc:.4f} | "
            f"Selected epoch: {best_epoch} | "
            f"Fold test accuracy: "
            f"{corresponding_test_acc:.4f}"
        )

        del model
        del optimizer
        del scaler
        del train_loader
        del train_eval_loader
        del test_loader
        del train_graphs
        del test_graphs
        del topo_train_raw
        del topo_test_raw
        del topo_train
        del topo_test

        gc.collect()

        if use_cuda:
            torch.cuda.empty_cache()

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = (
        float(np.std(fold_accuracies, ddof=1))
        if len(fold_accuracies) > 1
        else 0.0
    )

    return {
        "hidden_channels": hidden_channels,
        "lr": lr,
        "alpha": alpha,
        "dropout": args.dropout,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "fold_accuracies": fold_accuracies,
        "fold_best_train_accuracies": fold_best_train_accuracies,
        "fold_best_epochs": fold_best_epochs,
    }


def save_results(
    all_results: List[Dict],
    output_csv: str,
) -> None:
    rows = []

    for result in all_results:
        rows.append(
            {
                "hidden_channels": result["hidden_channels"],
                "lr": result["lr"],
                "alpha": result["alpha"],
                "dropout": result["dropout"],
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "fold_accuracies": str(
                    [
                        round(value, 6)
                        for value in result["fold_accuracies"]
                    ]
                ),
                "fold_best_train_accuracies": str(
                    [
                        round(value, 6)
                        for value in result[
                            "fold_best_train_accuracies"
                        ]
                    ]
                ),
                "fold_best_epochs": str(
                    result["fold_best_epochs"]
                ),
            }
        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    results_df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Graph-topology hyperparameter search with "
            "training-accuracy early stopping"
        )
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imdb-binary",
    )

    parser.add_argument(
        "--hidden_channels_list",
        type=int,
        nargs="+",
        default=[32, 64, 128],
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--alpha_list",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.2],
    )

    # Learning rates to search
    parser.add_argument(
        "--lr_list",
        type=float,
        nargs="+",
        default=[0.0001, 0.0005, 0.001, 0.005],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help=(
            "Minimum full-training-accuracy increase required "
            "to reset patience."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="hyperparameter_results.csv",
    )

    args = parser.parse_args()

    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.runs < 2:
        raise ValueError("--runs must be at least 2.")
    if not 0.0 < args.temperature:
        raise ValueError("--temperature must be positive.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError(
            "Dropout must satisfy 0 <= dropout < 1."
        )

    if any(value <= 0.0 for value in args.lr_list):
        raise ValueError(
            "Every learning rate must be greater than zero."
        )

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if args.device < 0 or args.device >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid CUDA device {args.device}; "
                f"{torch.cuda.device_count()} device(s) are available."
            )

        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"Using device: {device}")
        print(
            f"GPU: {torch.cuda.get_device_name(args.device)}"
        )
        print(
            f"CUDA AMP: "
            f"{'disabled' if args.no_amp else 'enabled'}"
        )
    else:
        device = torch.device("cpu")
        print("CUDA is unavailable; using CPU.")

    print("\nArguments:")
    print(args)

    print(f"\nProcessing dataset: {args.dataset}")
    dataset = load_data(args.dataset)

    if len(dataset) == 0:
        raise ValueError("The loaded dataset is empty.")
    # if dataset[0].x is None:
    #     raise ValueError(
    #         "dataset[0].x is None. Add node features before "
    #         "building GINEncoder."
    #     )

    print(dataset[0])

    list_hks, thresholds_hks, raw_labels = get_thresh_hks(
        dataset,
        10,
        0.1,
    )

    graph_features = []
    new_data_list = []

    for graph_id in tqdm(
        range(len(dataset)),
        desc="Computing topological features",
    ):
        topo_feature = get_Topo_Fe(
            dataset[graph_id],
            list_hks[graph_id],
            thresholds_hks,
        )

        topo_feature = torch.as_tensor(
            topo_feature,
            dtype=torch.float32,
        )
        graph_features.append(topo_feature)
        if args.dataset in ['imdb-binary', 'imdb-multi']:
            data = dataset[graph_id]
            deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
            data.x = deg
            new_data_list.append(data)

    topo_tensor = torch.stack(graph_features)

    if len(new_data_list) != 0:
        dataset = MyDataset(new_data_list)

    print(dataset[0])

    if torch.is_tensor(raw_labels):
        labels_original = raw_labels.long().view(-1)
    else:
        labels_original = torch.tensor(
            raw_labels,
            dtype=torch.long,
        ).view(-1)

    if len(labels_original) != len(dataset):
        raise ValueError(
            "The number of labels does not match the number "
            "of graphs."
        )

    # Ensure class labels are contiguous: 0, 1, ..., C-1.
    unique_labels, labels = torch.unique(
        labels_original,
        sorted=True,
        return_inverse=True,
    )

    node_feat_dim = int(dataset[0].x.shape[1])
    topo_dim = int(topo_tensor.shape[1])
    num_classes = int(unique_labels.numel())

    print(f"Number of graphs: {len(dataset)}")
    print(f"Node feature dimension: {node_feat_dim}")
    print(f"Topological feature dimension: {topo_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Original class values: {unique_labels.tolist()}")

    labels_numpy = labels.cpu().numpy()

    class_counts = np.bincount(labels_numpy)
    smallest_class = int(class_counts.min())

    if args.runs > smallest_class:
        raise ValueError(
            f"--runs={args.runs} is too large. The smallest "
            f"class has only {smallest_class} samples, so "
            f"StratifiedKFold requires runs <= {smallest_class}."
        )

    # Create the splits once so every hyperparameter combination
    # is evaluated on exactly the same folds.
    cross_validator = StratifiedKFold(
        n_splits=args.runs,
        shuffle=True,
        random_state=args.seed,
    )

    cv_splits = list(
        cross_validator.split(
            np.zeros(len(labels_numpy)),
            labels_numpy,
        )
    )

    hyperparameter_grid = list(
        itertools.product(
            args.hidden_channels_list,
            args.lr_list,
            args.alpha_list,
        )
    )

    print(
        f"\nRunning {len(hyperparameter_grid)} "
        "hyperparameter combinations over "
        f"{args.runs} folds."
    )

    all_results: List[Dict] = []

    for combination_number, (
            hidden_channels,
            lr,
            alpha,
    ) in enumerate(hyperparameter_grid, start=1):
        print("\n" + "=" * 76)
        print(
            f"Combination {combination_number}/"
            f"{len(hyperparameter_grid)}: "
            f"hidden_channels={hidden_channels}, "
            f"lr={lr}, alpha={alpha}, "
            f"dropout={args.dropout}"
        )
        print("=" * 76)

        result = run_hyperparameter_combination(
            args=args,
            dataset=dataset,
            topo_tensor=topo_tensor,
            labels=labels,
            cv_splits=cv_splits,
            hidden_channels=hidden_channels,
            lr=lr,
            alpha=alpha,
            node_feat_dim=node_feat_dim,
            topo_dim=topo_dim,
            num_classes=num_classes,
            device=device,
        )

        all_results.append(result)
        save_results(all_results, args.output_csv)

        print(
            "\nCombination result:"
            f"\n  Mean fold accuracy: "
            f"{result['mean_accuracy']:.4f} "
            f"± {result['std_accuracy']:.4f}"
            f"\n  Fold accuracies: "
            f"{[round(x, 4) for x in result['fold_accuracies']]}"
            f"\n  Best training accuracies: "
            f"{[round(x, 4) for x in result['fold_best_train_accuracies']]}"
            f"\n  Selected epochs: "
            f"{result['fold_best_epochs']}"
        )

    # Highest mean accuracy wins. Lower standard deviation breaks ties.
    best_result = sorted(
        all_results,
        key=lambda item: (
            -item["mean_accuracy"],
            item["std_accuracy"],
        ),
    )[0]

    print("\n" + "#" * 76)
    print("BEST HYPERPARAMETER CONFIGURATION")
    print("#" * 76)
    print(
        f"Hidden channels: "
        f"{best_result['hidden_channels']}"
    )
    print(f"Learning rate: {best_result['lr']}")
    print(f"Alpha: {best_result['alpha']}")
    print(f"Fixed dropout: {best_result['dropout']}")
    print(
        f"Best mean accuracy: "
        f"{best_result['mean_accuracy']:.4f} "
        f"± {best_result['std_accuracy']:.4f}"
    )
    print(
        "Fold accuracies:",
        [
            round(value, 4)
            for value in best_result["fold_accuracies"]
        ],
    )
    print(
        "Best training accuracies:",
        [
            round(value, 4)
            for value in best_result[
                "fold_best_train_accuracies"
            ]
        ],
    )
    print(
        "Selected epoch in each fold:",
        best_result["fold_best_epochs"],
    )
    print(f"\nAll results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
