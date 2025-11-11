import networkx as nx
import numpy as np
import pandas as pd
import pickle
import math
import pyflagser
import statistics
import argparse
from torch_geometric.utils import degree
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import time
from data_loader import load_data
from model import *
from modules import *
from logger import print_stat,stat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
# ---------- Train for one epoch ----------
def train_one_epoch(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    preds = out.argmax(dim=1)
    acc = (preds == y_train).float().mean().item()
    return loss.item(), acc


# ---------- Evaluate ----------
@torch.no_grad()
def evaluate(model, criterion, X_test, y_test):
    model.eval()
    out = model(X_test)
    loss = criterion(out, y_test)
    preds = out.argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
    return loss.item(), acc


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='MLP Graph Classification')
    parser.add_argument('--dataset', type=str, default='imdb-multi')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    # dataset='REDDIT-MULTI-5K'#MUTAG,PROTEINS,BRZ,IMDB-BINARY,COX2,IMDB-MULTI,REDDIT-BINARY,REDDIT-MULTI-5K
    from sklearn.model_selection import train_test_split
    print(f"Processing dataset: {args.dataset}")

    # Load features and labels
    dataset = load_data(args.dataset)
    print(dataset[0])

    list_hks, thres_hks, label = get_thresh_hks(dataset, 10, 0.1)
    list_deg, thres_deg = get_thresh(dataset, 10)
    graph_features = []
    new_data_list = []
    for graph_id in tqdm(range(len(dataset))):
        topo_fe = get_Topo_Fe(dataset[graph_id], list_hks[graph_id], thres_hks)
        # Make sure it's a torch tensor
        topo_fe = torch.tensor(topo_fe, dtype=torch.float)
        graph_features.append(topo_fe)
        if args.dataset in ['imdb-binary', 'imdb-multi']:
            data = dataset[graph_id]
            deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
            data.x = deg
            new_data_list.append(data)
        # dataset[graph_id]=data
        # dataset[graph_id].x=torch.eye(dataset[graph_id].num_nodes)
    X = torch.stack(graph_features)
    y = torch.tensor(label, dtype=torch.long)

    X, y = X.to(device), y.to(device)

    print(f"Running {args.folds}-fold cross-validation on {len(y)} samples")

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MLP(X.shape[1], args.hidden_dim, len(torch.unique(y)), args.num_layers, args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in tqdm(range(1, args.epochs + 1)):
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, X_train, y_train)
            test_loss, test_acc = evaluate(model, criterion, X_test, y_test)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        acc = print_stat(train_accuracies, test_accuracies)
        fold_results.append(acc)

    print("\n=== Cross-validation Results ===")
    print(f"Mean Test Accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == "__main__":
    main()
