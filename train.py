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

class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)
class GraphWithTopoDataset(Dataset):
    def __init__(self, graphs, topo_feats, labels):
        self.graphs = graphs
        self.topo_feats = topo_feats
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.topo_feats[idx], self.labels[idx]

# def contrastive_loss(z1, z2, temperature=0.1):
#     z1 = F.normalize(z1, dim=-1)
#     z2 = F.normalize(z2, dim=-1)
#     batch_size, _ = z1.shape
#     z = torch.cat([z1, z2], dim=0)  # [2B, D]
#     sim = torch.mm(z, z.t()) / temperature
#     labels = torch.arange(batch_size, device=z.device)
#     labels = torch.cat([labels, labels], dim=0)
#
#     mask = torch.eye(2*batch_size, device=z.device).bool()
#     sim = sim.masked_fill(mask, -9e15)
#
#     loss = F.cross_entropy(sim, labels)
#     return loss

def contrastive_loss(z1, z2, temperature=0.1):
    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch_size = z1.size(0)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(torch.cat([z1, z2], dim=0),
                              torch.cat([z1, z2], dim=0).t()) / temperature
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # Labels: positives are diagonal shifted by batch_size
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size, device=z1.device),
        torch.arange(0, batch_size, device=z1.device)
    ])
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

from torch_geometric.data import Batch
import torch.nn.functional as F

def train(model, loader, optimizer, device, alpha=0.1):
    model.train()
    total_loss, total_correct, total_examples = 0, 0, 0

    for batch_graphs, batch_topo, batch_labels in loader:
        # Batch graphs
        # batch_graphs = Batch.from_data_list(batch_graphs).to(device)
        batch_graphs = batch_graphs.to(device)

        # Move topo feats & labels
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits, proj, _ = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_topo)

        # Supervised classification loss
        cls_loss = F.cross_entropy(logits, batch_labels)

        # Contrastive loss between GIN and topo embeddings
        g_emb = model.gin_encoder(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
        t_emb = model.topo_encoder(batch_topo)
        con_loss = contrastive_loss(g_emb, t_emb)
        # print("CLS Loss:", cls_loss.item())
        # print("Contrastive Loss:", con_loss.item())

        loss = cls_loss + alpha * con_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_graphs.num_graphs
        total_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        total_examples += batch_graphs.num_graphs

    return total_loss / total_examples, total_correct / total_examples
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_examples = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_graphs, batch_topo, batch_labels in loader:
            # batch_graphs = Batch.from_data_list(batch_graphs).to(device)
            batch_graphs = batch_graphs.to(device)
            batch_topo = batch_topo.to(device)
            batch_labels = batch_labels.to(device)

            logits, _, _ = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_topo)
            preds = logits.argmax(dim=-1)

            total_correct += (preds == batch_labels).sum().item()
            total_examples += batch_graphs.num_graphs

            y_true.append(batch_labels.cpu())
            y_pred.append(preds.cpu())

    acc = total_correct / total_examples
    return acc, torch.cat(y_true), torch.cat(y_pred)


def main():
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='imdb-multi')#MUTAG,PROTEINS,BZR,IMDB-BINARY,COX2,IMDB-MULTI,REDDIT-BINARY,REDDIT-MULTI-5K
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gap_pmeter', type=int, default=2)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    #dataset='REDDIT-MULTI-5K'#MUTAG,PROTEINS,BRZ,IMDB-BINARY,COX2,IMDB-MULTI,REDDIT-BINARY,REDDIT-MULTI-5K
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
        if args.dataset in ['imdb-binary','imdb-multi']:
            data=dataset[graph_id]
            deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
            data.x = deg
            new_data_list.append(data)
        #dataset[graph_id]=data
        #dataset[graph_id].x=torch.eye(dataset[graph_id].num_nodes)
    topo_tensor = torch.stack(graph_features)
    if len(new_data_list) !=0:
        dataset = MyDataset(new_data_list)
    print(dataset[0])

    y = torch.tensor(label, dtype=torch.long)

    # Convert to PyTorch tensors

    # Extract dataset details
    num_samples = len(y)
    num_features1 = len(topo_tensor[0])
    num_classes = len(np.unique(y))

    # Define input and output dimensions
    node_feat_dim=dataset[0].x.shape[1]
    topo_dim=num_features1
    hidden_dim = args.hidden_channels
    output_dim = num_classes
    n_heads = args.head
    n_layers = args.num_layers


    # K-Fold Cross Validation
    kfold = KFold(n_splits=args.runs, shuffle=True,random_state=42)
    acc_per_fold = []
    fold_no = 1

    for train_idx, test_idx in kfold.split(topo_tensor):
        # Split data
        X1_train, X1_test = topo_tensor[train_idx], topo_tensor[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        data_train = [dataset[i] for i in train_idx]
        data_test = [dataset[i] for i in test_idx]

        train_dataset = GraphWithTopoDataset(data_train, X1_train, y_train)
        test_dataset = GraphWithTopoDataset(data_test, X1_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


        # Define encoders
        gin_encoder = GINEncoder(in_dim=node_feat_dim, hidden_dim=64, out_dim=128)
        topo_encoder = MLPEncoder(in_dim=topo_dim, hidden_dim=64, out_dim=128)

        # Initialize model, loss function, and optimizer
        # Model
        model = HybridGraphTopoModel(gin_encoder, topo_encoder, hidden_dim=128, proj_dim=64, num_classes=num_classes)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_losses = []
        train_accuracies = []
        test_accuracies = []

        # Training loop
        for epoch in range(1, args.epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, device, alpha=0.1)
            val_acc, y_true, y_pred = evaluate(model, test_loader, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(val_acc)
            print(
                f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")
        print(f'Score for fold {fold_no}: ')
        acc = print_stat(train_accuracies, test_accuracies)
        acc_per_fold.append(acc)
        fold_no += 1
    print(acc_per_fold)
    stat(acc_per_fold, 'accuracy')
if __name__ == "__main__":
    main()