import argparse
import random
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from models import *


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
# Task inference
# =========================================================
def infer_problem_type(dataset):
    task_type = dataset.task_type.lower().strip()

    # OGB graph property datasets often use binary or multi-task binary
    # classification setups for molecular benchmarks.
    if "binary" in task_type:
        return "multi-label"

    if dataset.num_tasks > 1:
        return "multi-label"

    return "single-label"


# =========================================================
# Build graph model
# Assumes each model returns graph-level logits
# =========================================================
def build_graph_model(model_type, in_dim, hidden_dim, out_dim,
                      num_layers, dropout, heads=4):
    model_type = model_type.lower()

    if model_type == "gcn":
        return GCNGraphClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_type == "gin":
        return GINGraphClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_type == "graphsage":
        return GraphSAGEGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_type == "gat":
        return GATGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    elif model_type == "gps":
        return GPSGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    elif model_type == "sgformer":
        return SGFormerGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =========================================================
# Classification loss
# =========================================================
def compute_cls_loss(logits, labels, problem_type):
    if problem_type == "single-label":
        labels = labels.view(-1).long()
        return F.cross_entropy(logits, labels)

    elif problem_type == "multi-label":
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)

        labels = labels.float()
        mask = ~torch.isnan(labels)

        loss_mat = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
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

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)

        labels = batch.y
        if labels.ndim == 3 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        loss = compute_cls_loss(logits, labels, problem_type)
        loss.backward()
        optimizer.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_examples += bs

        if problem_type == "single-label":
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels.view(-1).long()).sum().item()

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

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y

        if labels.ndim == 3 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        if problem_type == "single-label":
            pred = logits.argmax(dim=-1, keepdim=True)
            true = labels.view(-1, 1)
            y_true.append(true.cpu())
            y_pred.append(pred.cpu())
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            y_true.append(labels.cpu())
            y_pred.append(logits.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Graph-only baselines on OGBG datasets"
    )

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-molsider")

    parser.add_argument(
        "--model_type",
        type=str,
        default="sgformer",
        choices=["gcn", "gin", "graphsage", "gat", "gps", "sgformer"],
        help="baseline graph model type"
    )

    parser.add_argument("--drop_ratio", type=float, default=0.5)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--gat_heads", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--filename", type=str, default="")

    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    # -----------------------------
    # Load dataset and evaluator
    # -----------------------------
    dataset = PygGraphPropPredDataset(name=args.dataset)
    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    print("Dataset:", args.dataset)
    print("Task type:", dataset.task_type)
    print("Eval metric:", dataset.eval_metric)
    print("Num tasks:", dataset.num_tasks)
    print("Example label:", dataset[0].y)

    problem_type = infer_problem_type(dataset)
    print("Inferred problem type:", problem_type)

    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    node_feat_dim = dataset[0].x.shape[1]

    if problem_type == "single-label":
        train_y = []
        for data in train_dataset:
            yy = data.y
            if not torch.is_tensor(yy):
                yy = torch.tensor(yy)
            train_y.append(yy.view(-1))
        train_y = torch.cat(train_y, dim=0)
        num_classes = int(train_y.max().item()) + 1
    else:
        num_classes = dataset.num_tasks

    print("node_feat_dim:", node_feat_dim)
    print("num_classes / out_dim:", num_classes)

    metric_name = dataset.eval_metric

    all_train_scores = []
    all_val_scores = []
    all_test_scores = []

    for run in range(args.runs):
        run_seed = args.seed + run
        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{args.runs} | Seed = {run_seed}")
        print(f"{'=' * 60}")

        set_seed(run_seed)
        start_time = time.time()

        g = torch.Generator()
        g.manual_seed(run_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=g
        )
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

        model = build_graph_model(
            model_type=args.model_type,
            in_dim=node_feat_dim,
            hidden_dim=args.emb_dim,
            out_dim=num_classes,
            num_layers=args.num_layer,
            dropout=args.drop_ratio,
            heads=args.gat_heads
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_curve = []
        valid_curve = []
        test_curve = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(
                model, train_loader, optimizer, device,
                problem_type=problem_type
            )

            train_perf = evaluate(model, train_loader, device, evaluator, problem_type)
            valid_perf = evaluate(model, valid_loader, device, evaluator, problem_type)
            test_perf = evaluate(model, test_loader, device, evaluator, problem_type)

            # print(
            #     f"Run {run + 1:02d}, Epoch {epoch:03d}, "
            #     f"Loss {train_loss:.4f}, "
            #     f"Train {train_perf[metric_name]:.4f}, "
            #     f"Val {valid_perf[metric_name]:.4f}, "
            #     f"Test {test_perf[metric_name]:.4f}"
            # )

            train_curve.append(train_perf[metric_name])
            valid_curve.append(valid_perf[metric_name])
            test_curve.append(test_perf[metric_name])

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
        end_time = time.time()  # ⬅️ END TIMER
        run_time = end_time - start_time
        print(f"Runtime for run {run + 1}: {run_time:.2f} seconds")

    print("\nFinished all runs!")
    print(f"Train: {np.mean(all_train_scores):.4f} ± {np.std(all_train_scores):.4f}")
    print(f"Valid: {np.mean(all_val_scores):.4f} ± {np.std(all_val_scores):.4f}")
    print(f"Test:  {np.mean(all_test_scores):.4f} ± {np.std(all_test_scores):.4f}")

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
            "ModelType": args.model_type,
            "ProblemType": problem_type,
            "Metric": metric_name,
            "Runs": args.runs
        }, args.filename)


if __name__ == "__main__":
    main()