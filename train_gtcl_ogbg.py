import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# your existing imports
from models import *
from modules import *
from data_loader import *
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: more reproducible but sometimes slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# Dataset wrapper
# =========================================================
class GraphWithTopoDataset(Dataset):
    def __init__(self, graphs, topo_feats, labels):
        self.graphs = graphs
        self.topo_feats = topo_feats
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.topo_feats[idx], self.labels[idx]


# =========================================================
# Contrastive loss
# =========================================================
def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim_matrix = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z1.device),
        torch.arange(0, batch_size, device=z1.device)
    ])

    return F.cross_entropy(sim_matrix, labels)


# =========================================================
# Task inference
# =========================================================
def infer_problem_type(dataset):
    """
    Returns one of:
        - 'single-label'
        - 'multi-label'
    """
    task_type = dataset.task_type.lower().strip()

    # OGB molecular datasets are typically binary classification or
    # multi-task binary classification -> BCE-style
    if "binary" in task_type:
        if dataset.num_tasks == 1:
            # still BCE
            return "multi-label"
        return "multi-label"

    # multi-task classification -> treat as multi-label BCE
    if dataset.num_tasks > 1:
        return "multi-label"

    # otherwise assume single-label classification
    return "single-label"


# =========================================================
# Topology encoder
# =========================================================

# =========================================================


def build_graph_encoder(model_type, in_dim, hidden_dim, out_dim, num_layers, dropout, heads=4):
    model_type = model_type.lower()

    if model_type == "gin":
        return GINGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_type == "gcn":
        return GCNGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_type == "graphsage":
        return GraphSAGEGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif model_type == "gat":
        return GATGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, heads, dropout)
    elif model_type == "gps":
        return GPSGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, heads, dropout)
    elif model_type == "sgformer":
        return SGFormerGraphEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =========================================================
# Hybrid model
# =========================================================



# =========================================================
# Classification loss
# =========================================================
# def compute_cls_loss(logits, labels, problem_type):
#     if problem_type == "single-label":
#         labels = labels.view(-1).long()
#         return F.cross_entropy(logits, labels)
#
#     elif problem_type == "multi-label":
#         # labels: [B, num_tasks]
#         if labels.ndim == 1:
#             labels = labels.unsqueeze(-1)
#
#         labels = labels.float()
#         mask = ~torch.isnan(labels)
#
#         loss_mat = F.binary_cross_entropy_with_logits(
#             logits, labels, reduction="none"
#         )
#         loss = (loss_mat * mask.float()).sum() / mask.float().sum()
#         return loss
#
#     else:
#         raise ValueError(f"Unknown problem_type: {problem_type}")

def compute_cls_loss(logits, labels, problem_type):
    if problem_type == "single-label":
        labels = labels.view(-1).long()
        return F.cross_entropy(logits, labels)

    elif problem_type == "multi-label":
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)

        labels = labels.float()
        mask = ~torch.isnan(labels)

        if mask.sum() == 0:
            return logits.sum() * 0.0

        # Important: remove NaN before BCE
        labels_clean = torch.where(mask, labels, torch.zeros_like(labels))

        loss_mat = F.binary_cross_entropy_with_logits(
            logits, labels_clean, reduction="none"
        )

        loss = (loss_mat * mask.float()).sum() / mask.float().sum()
        return loss

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
# =========================================================
# Train / Evaluate
# =========================================================
def train(model, loader, optimizer, device, alpha=0.1, problem_type="multi-label"):
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch_graphs, batch_topo, batch_labels in loader:
        batch_graphs = batch_graphs.to(device)
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        optimizer.zero_grad()

        logits, g_proj, t_proj = model(
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.batch,
            batch_topo
        )

        cls_loss = compute_cls_loss(logits, batch_labels, problem_type)
        con_loss = contrastive_loss(g_proj, t_proj)
        loss = cls_loss + alpha * con_loss

        loss.backward()
        optimizer.step()

        bs = batch_graphs.num_graphs
        total_loss += loss.item() * bs
        total_examples += bs

        if problem_type == "single-label":
            pred = logits.argmax(dim=-1)
            total_correct += (pred == batch_labels.view(-1).long()).sum().item()

    train_acc = None
    if problem_type == "single-label":
        train_acc = total_correct / total_examples

    return total_loss / total_examples, train_acc


@torch.no_grad()
def evaluate(model, loader, device, evaluator, dataset, problem_type="multi-label"):
    model.eval()

    y_true = []
    y_pred = []

    for batch_graphs, batch_topo, batch_labels in loader:
        batch_graphs = batch_graphs.to(device)
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        logits, _, _ = model(
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.batch,
            batch_topo
        )

        if problem_type == "single-label":
            pred = logits.argmax(dim=-1, keepdim=True)
            true = batch_labels.view(-1, 1)
            y_true.append(true.cpu())
            y_pred.append(pred.cpu())
        else:
            if batch_labels.ndim == 1:
                batch_labels = batch_labels.unsqueeze(-1)
            y_true.append(batch_labels.cpu())
            y_pred.append(logits.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()


    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Hybrid graph+topology model on OGBG datasets')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='ogbg-molsider')

    parser.add_argument('--model_type', type=str, default='sgformer',
                        choices=['gin', 'graphsage', 'gat', 'gps', 'sgformer'],
                        help='graph encoder type')

    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--topo_hidden_dim', type=int, default=128)

    parser.add_argument('--gat_heads', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--feature', type=str, default="full")
    parser.add_argument('--filename', type=str, default="")
    parser.add_argument('--runs', type=int, default=5,
                        help='number of independent runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='base random seed')

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = PygGraphPropPredDataset(name=args.dataset)
    evaluator = Evaluator(args.dataset)

    print("Dataset:", args.dataset)
    print("Task type:", dataset.task_type)
    print("Eval metric:", dataset.eval_metric)
    print("Num tasks:", dataset.num_tasks)
    print("Example label:", dataset[0].y)

    problem_type = infer_problem_type(dataset)
    print("Inferred problem type:", problem_type)


    # -----------------------------
    # Build topo features
    start_feat = time.time()
    # -----------------------------
    list_hks, thres_hks, label = get_thresh_hks(dataset, 10, 0.1)
    list_deg, thres_deg = get_thresh(dataset, 10)

    graph_features = []
    # for graph_id in tqdm(range(len(dataset)), desc="Building topo features"):
    for graph_id in range(len(dataset)):
        topo_fe = get_Topo_Fe(dataset[graph_id], list_hks[graph_id], thres_hks)
        topo_fe = torch.tensor(topo_fe, dtype=torch.float)
        graph_features.append(topo_fe)

    topo_tensor = torch.stack(graph_features)
    print(topo_tensor[0])
    end_feat = time.time()
    feature_time = end_feat - start_feat

    print(f"Feature extraction time: {feature_time:.3f} seconds")

    # Better to use dataset labels directly
    y = []
    for i in range(len(dataset)):
        yy = dataset[i].y
        if not torch.is_tensor(yy):
            yy = torch.tensor(yy)
        y.append(yy.view(-1))
    y = torch.stack(y).float()

    # -----------------------------
    # Split
    # -----------------------------
    split_idx = dataset.get_idx_split()

    X_train = topo_tensor[split_idx["train"]]
    X_val   = topo_tensor[split_idx["valid"]]
    X_test  = topo_tensor[split_idx["test"]]

    y_train = y[split_idx["train"]]
    y_val   = y[split_idx["valid"]]
    y_test  = y[split_idx["test"]]

    data_train = [dataset[i] for i in split_idx["train"]]
    data_valid = [dataset[i] for i in split_idx["valid"]]
    data_test  = [dataset[i] for i in split_idx["test"]]

    train_dataset = GraphWithTopoDataset(data_train, X_train, y_train)
    valid_dataset = GraphWithTopoDataset(data_valid, X_val, y_val)
    test_dataset  = GraphWithTopoDataset(data_test, X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # -----------------------------
    # Dimensions
    # -----------------------------
    node_feat_dim = dataset[0].x.shape[1]
    topo_dim = topo_tensor.shape[1]

    if problem_type == "single-label":
        # number of classes from labels
        all_train_labels = y_train.view(-1)
        num_classes = int(all_train_labels.max().item()) + 1
    else:
        num_classes = dataset.num_tasks

    print("node_feat_dim:", node_feat_dim)
    print("topo_dim:", topo_dim)
    print("num_classes / out_dim:", num_classes)
    all_val_scores = []
    all_test_scores = []
    all_train_scores = []

    metric_name = dataset.eval_metric

    for run in range(args.runs):
        run_seed = args.seed + run
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{args.runs} | Seed = {run_seed}")
        print(f"{'=' * 60}")

        set_seed(run_seed)

        graph_encoder = build_graph_encoder(
            model_type=args.model_type,
            in_dim=node_feat_dim,
            hidden_dim=args.emb_dim,
            out_dim=args.emb_dim,
            num_layers=args.num_layer,
            dropout=args.drop_ratio,
            heads=args.gat_heads
        )

        topo_encoder = MLPEncoder(
            in_dim=topo_dim,
            hidden_dim=args.topo_hidden_dim,
            out_dim=args.emb_dim
        )

        model = HybridGraphTopoModel(
            graph_encoder=graph_encoder,
            topo_encoder=topo_encoder,
            hidden_dim=args.emb_dim,
            proj_dim=args.proj_dim,
            out_dim=num_classes
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_curve = []
        valid_curve = []
        test_curve = []
        start_train = time.time()

        for epoch in range(1, args.epochs + 1):
            #print(f"\n===== Run {run + 1}, Epoch {epoch}")

            train_loss, train_acc = train(
                model, train_loader, optimizer, device,
                alpha=args.alpha,
                problem_type=problem_type
            )

            train_perf = evaluate(model, train_loader, device, evaluator, dataset, problem_type)
            valid_perf = evaluate(model, valid_loader, device, evaluator, dataset, problem_type)
            test_perf = evaluate(model, test_loader, device, evaluator, dataset, problem_type)

            # print(f"Train loss: {train_loss:.4f}")
            # if train_acc is not None:
            #     print(f"Train acc: {train_acc:.4f}")
            #
            # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[metric_name])
            valid_curve.append(valid_perf[metric_name])
            test_curve.append(test_perf[metric_name])
        end_train = time.time()
        train_time = end_train - start_train

        print(f"Training time (run {run + 1}): {train_time:.2f} seconds")

        best_val_epoch = int(np.argmax(np.array(valid_curve)))
        best_train = train_curve[best_val_epoch]
        best_val = valid_curve[best_val_epoch]
        best_test = test_curve[best_val_epoch]

        all_train_scores.append(best_train)
        all_val_scores.append(best_val)
        all_test_scores.append(best_test)

        print(f"\nRun {run + 1} finished")
        print(f"Best epoch: {best_val_epoch + 1}")
        print(f"Best validation score: {best_val:.4f}")
        print(f"Test score: {best_test:.4f}")

    print('\nFinished all runs!')
    print(f'Train: {np.mean(all_train_scores):.4f} ± {np.std(all_train_scores):.4f}')
    print(f'Valid: {np.mean(all_val_scores):.4f} ± {np.std(all_val_scores):.4f}')
    print(f'Test: {np.mean(all_test_scores):.4f} ± {np.std(all_test_scores):.4f}')

    if args.filename != '':
        torch.save({
            'TrainScores': all_train_scores,
            'ValScores': all_val_scores,
            'TestScores': all_test_scores,
            'TrainMean': float(np.mean(all_train_scores)),
            'TrainStd': float(np.std(all_train_scores)),
            'ValMean': float(np.mean(all_val_scores)),
            'ValStd': float(np.std(all_val_scores)),
            'TestMean': float(np.mean(all_test_scores)),
            'TestStd': float(np.std(all_test_scores)),
            'ModelType': args.model_type,
            'ProblemType': problem_type,
            'Metric': metric_name,
            'Runs': args.runs
        }, args.filename)
if __name__ == "__main__":
    main()