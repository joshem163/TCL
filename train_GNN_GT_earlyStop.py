import argparse
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.model_selection import KFold, train_test_split

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
from data_loader import load_data
from models import *
from modules import *
from logger import print_stat,stat
from torch_geometric.data import InMemoryDataset, Data

class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)

# -----------------------------
# Utilities
# -----------------------------
def ensure_node_features(dataset):
    """Ensure each graph has node features. If absent, use constant ones."""
    if dataset.num_features == 0:
        new_data=[]
        for data in dataset:
            deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
            data.x = deg
            new_data.append(data)
        in_dim = new_data[0].x.shape[1]
        dataset = MyDataset(new_data)
    else:
        in_dim = dataset.num_features
    return dataset,in_dim

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)
    return correct / total if total > 0 else 0.0

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == data.y).sum())
        total += data.y.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = total_correct / total if total > 0 else 0.0
    return avg_loss, acc


# -----------------------------
# Main (10-fold CV)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="GIN on TU Datasets with 10-fold CV")
    parser.add_argument("--dataset", type=str, default="ptc",
                        help="TU dataset name, e.g., MUTAG, BZR, PROTEINS, IMDB-BINARY, IMDB-MULTI")
    parser.add_argument("--model", type=str, default="graphomer",
                        help="gps, sgformer,graphomer")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_data(args.dataset)

    # Ensure node features exist
    dataset,in_dim = ensure_node_features(dataset)
    num_classes = dataset.num_classes

    # Labels for stratification
    y_all = np.array([int(d.y.item()) for d in dataset])

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_test_accs = []

    print(f"Dataset: {args.dataset} | Graphs: {len(dataset)} | Classes: {num_classes} | In-dim: {in_dim}")
    print(f"Running {args.folds}-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset))), start=1):
        # Split dataset into train/test for this fold
        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model=='gin':
            model = GINEncoder(in_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
        elif args.model== 'gps':
            model = GPSGraphEncoder(in_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
        elif args.model== 'sgformer':
            model = SGFormerGraphEncoder(in_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
        elif args.model== 'graphomer':
            model = Graphormer(in_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
        else:
            print("model does not exist")
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_train_acc = -1.0
        best_test_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0

        train_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in trange(
                1,
                args.epochs + 1,
                desc=f"Fold {fold}",
                leave=False
        ):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device
            )

            test_acc = evaluate(
                model,
                test_loader,
                device
            )

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Early stopping based on training accuracy
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_test_acc = test_acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f} | "
                    f"Best Train Acc: {best_train_acc:.4f} | "
                    f"Test@Best Train: {best_test_acc:.4f} | "
                    f"Patience: {epochs_no_improve}/{args.patience}"
                )

            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best training accuracy occurred at epoch {best_epoch}."
                )
                break

        # Use the test accuracy corresponding to the best training accuracy
        acc = best_test_acc
        fold_test_accs.append(acc)

        print(
            f"Fold {fold} | "
            f"Best Train Acc: {best_train_acc:.4f} | "
            f"Best Epoch: {best_epoch} | "
            f"Corresponding Test Acc: {best_test_acc:.4f}"
        )
    mean_acc = float(np.mean(fold_test_accs))
    std_acc = float(np.std(fold_test_accs))
    print("\n========== 10-Fold CV Results ==========")
    print(f"Mean Test Acc: {100*mean_acc:.2f} ± {100*std_acc:.2f}")


if __name__ == "__main__":
    main()
