import argparse
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from models import *
from modules import *
from data_loader import *


# =========================================================
# Seed
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Dataset wrapper for topological features
# =========================================================
class TopoFeatureDataset(Dataset):
    def __init__(self, topo_feats, labels):
        self.topo_feats = topo_feats
        self.labels = labels

    def __len__(self):
        return len(self.topo_feats)

    def __getitem__(self, idx):
        return self.topo_feats[idx], self.labels[idx]


# =========================================================
# Your MLP
# =========================================================
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, problem_type="multi-label"):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.problem_type = problem_type

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        if self.problem_type == "single-label":
            return torch.log_softmax(x, dim=-1)

        return x


# =========================================================
# Task inference
# =========================================================
def infer_problem_type(dataset):
    task_type = dataset.task_type.lower().strip()

    if "binary" in task_type:
        return "multi-label"

    if dataset.num_tasks > 1:
        return "multi-label"

    return "single-label"


# =========================================================
# Classification loss
# =========================================================
def compute_cls_loss(logits, labels, problem_type):
    if problem_type == "single-label":
        labels = labels.view(-1).long()
        return F.nll_loss(logits, labels)

    elif problem_type == "multi-label":
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)

        labels = labels.float()
        mask = ~torch.isnan(labels)

        if mask.sum() == 0:
            return logits.sum() * 0.0

        labels_clean = torch.where(mask, labels, torch.zeros_like(labels))

        loss_mat = F.binary_cross_entropy_with_logits(
            logits,
            labels_clean,
            reduction="none"
        )

        loss = (loss_mat * mask.float()).sum() / mask.float().sum()
        return loss

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


# =========================================================
# Train
# =========================================================
def train(model, loader, optimizer, device, problem_type="multi-label"):
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch_topo, batch_labels in loader:
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        optimizer.zero_grad()

        logits = model(batch_topo)
        loss = compute_cls_loss(logits, batch_labels, problem_type)

        loss.backward()
        optimizer.step()

        bs = batch_topo.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

        if problem_type == "single-label":
            pred = logits.argmax(dim=-1)
            total_correct += (
                pred == batch_labels.view(-1).long()
            ).sum().item()

    train_acc = None
    if problem_type == "single-label":
        train_acc = total_correct / total_examples

    return total_loss / total_examples, train_acc


# =========================================================
# Evaluate
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device, evaluator, problem_type="multi-label"):
    model.eval()

    y_true = []
    y_pred = []

    for batch_topo, batch_labels in loader:
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        logits = model(batch_topo)

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

    input_dict = {
        "y_true": y_true,
        "y_pred": y_pred
    }

    return evaluator.eval(input_dict)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="MLP on topological features for OGBG datasets"
    )

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-moltoxcast")

    parser.add_argument("--drop_ratio", type=float, default=0.5)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--feature", type=str, default="full")
    parser.add_argument("--filtration", type=str, default="hks")

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--filename", type=str, default="")

    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

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

    metric_name = dataset.eval_metric

    # -----------------------------
    # Build topological features
    # -----------------------------
    start_feat = time.time()

    if args.filtration == "hks":
        list_node_attri, thres, label = get_thresh_hks(dataset, 10, 0.1)
    elif args.filtration == "atom":
        list_node_attri, thres, label = get_thresh_atomic_feature(dataset, 10)
    elif args.filtration == "atomN":
        list_node_attri, thres, label = get_thres_atom(dataset, 10)
    else:
        raise ValueError(f"Unknown filtration: {args.filtration}")

    graph_features = []

    for graph_id in range(len(dataset)):
        topo_fe = get_Topo_Fe(
            dataset[graph_id],
            list_node_attri[graph_id],
            thres
        )
        topo_fe = torch.tensor(topo_fe, dtype=torch.float)
        graph_features.append(topo_fe)

    topo_tensor = torch.stack(graph_features)

    end_feat = time.time()
    feature_time = end_feat - start_feat

    print(f"PH feature extraction time: {feature_time:.3f} seconds")

    # -----------------------------
    # Labels
    # -----------------------------
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
    X_val = topo_tensor[split_idx["valid"]]
    X_test = topo_tensor[split_idx["test"]]

    y_train = y[split_idx["train"]]
    y_val = y[split_idx["valid"]]
    y_test = y[split_idx["test"]]

    train_dataset = TopoFeatureDataset(X_train, y_train)
    valid_dataset = TopoFeatureDataset(X_val, y_val)
    test_dataset = TopoFeatureDataset(X_test, y_test)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # -----------------------------
    # Dimensions
    # -----------------------------
    topo_dim = topo_tensor.shape[1]

    if problem_type == "single-label":
        all_train_labels = y_train.view(-1)
        num_classes = int(all_train_labels.max().item()) + 1
    else:
        num_classes = dataset.num_tasks

    print("topo_dim:", topo_dim)
    print("num_classes / out_dim:", num_classes)

    all_train_scores = []
    all_val_scores = []
    all_test_scores = []
    all_train_times = []

    # =====================================================
    # Runs
    # =====================================================
    for run in range(args.runs):
        run_seed = args.seed + run

        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{args.runs} | Seed = {run_seed}")
        print(f"{'=' * 60}")

        set_seed(run_seed)

        g = torch.Generator()
        g.manual_seed(run_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=g
        )

        model = MLP(
            in_channels=topo_dim,
            hidden_channels=args.hidden_dim,
            out_channels=num_classes,
            num_layers=args.num_layer,
            dropout=args.drop_ratio,
            problem_type=problem_type
        ).to(device)

        model.reset_parameters()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_curve = []
        valid_curve = []
        test_curve = []

        start_train = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(
                model,
                train_loader,
                optimizer,
                device,
                problem_type=problem_type
            )

            train_perf = evaluate(
                model,
                train_loader,
                device,
                evaluator,
                problem_type=problem_type
            )

            valid_perf = evaluate(
                model,
                valid_loader,
                device,
                evaluator,
                problem_type=problem_type
            )

            test_perf = evaluate(
                model,
                test_loader,
                device,
                evaluator,
                problem_type=problem_type
            )

            train_score = train_perf[metric_name]
            valid_score = valid_perf[metric_name]
            test_score = test_perf[metric_name]

            # print(
            #     f"Run {run + 1:02d}, Epoch {epoch:03d}, "
            #     f"Loss {train_loss:.4f}, "
            #     f"Train {train_score:.4f}, "
            #     f"Val {valid_score:.4f}, "
            #     f"Test {test_score:.4f}"
            # )

            train_curve.append(train_score)
            valid_curve.append(valid_score)
            test_curve.append(test_score)

        end_train = time.time()
        train_time = end_train - start_train
        all_train_times.append(train_time)

        best_val_epoch = int(np.argmax(np.array(valid_curve)))

        best_train = train_curve[best_val_epoch]
        best_val = valid_curve[best_val_epoch]
        best_test = test_curve[best_val_epoch]

        all_train_scores.append(best_train*100)
        all_val_scores.append(best_val*100)
        all_test_scores.append(best_test*100)

        print(f"\nRun {run + 1} finished")
        print(f"Best epoch: {best_val_epoch + 1}")
        print(f"Best train score: {best_train:.4f}")
        print(f"Best validation score: {best_val:.4f}")
        print(f"Test score: {best_test:.4f}")
        print(f"Training time: {train_time:.2f} seconds")

    # =====================================================
    # Final summary
    # =====================================================
    print("\nFinished all runs!")
    print(f"Train: {np.mean(all_train_scores):.4f} ± {np.std(all_train_scores):.4f}")
    print(f"Valid: {np.mean(all_val_scores):.4f} ± {np.std(all_val_scores):.4f}")
    print(f"Test:  {np.mean(all_test_scores):.4f} ± {np.std(all_test_scores):.4f}")

    print(f"PH feature extraction time: {feature_time:.2f} seconds")
    print(f"Training time: {np.mean(all_train_times):.2f} ± {np.std(all_train_times):.2f} seconds")
    print(
        f"Total time: "
        f"{feature_time + np.mean(all_train_times):.2f} seconds "
        f"(PH + mean training time)"
    )

    if args.filename != "":
        torch.save({
            "TrainScores": all_train_scores,
            "ValScores": all_val_scores,
            "TestScores": all_test_scores,

            "TrainMean": float(np.mean(all_train_scores)),
            "TrainStd": float(np.std(all_train_scores)),
            "ValMean": float(np.mean(all_val_scores)),
            "ValStd": float(np.std(all_val_scores)),
            "TestMean": float(np.mean(all_test_scores)),
            "TestStd": float(np.std(all_test_scores)),

            "PHFeatureTime": float(feature_time),
            "TrainTimes": all_train_times,
            "TrainTimeMean": float(np.mean(all_train_times)),
            "TrainTimeStd": float(np.std(all_train_times)),
            "TotalTimeMean": float(feature_time + np.mean(all_train_times)),

            "ModelType": "TopoMLP",
            "ProblemType": problem_type,
            "Metric": metric_name,
            "Dataset": args.dataset,
            "Filtration": args.filtration,
            "Runs": args.runs,
            "Seed": args.seed
        }, args.filename)


if __name__ == "__main__":
    main()