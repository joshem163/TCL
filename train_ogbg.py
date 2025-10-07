import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from model import *
from modules import *
from tqdm import tqdm
from data_loader import *
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()

class GraphWithTopoDataset(Dataset):
    def __init__(self, graphs, topo_feats, labels):
        self.graphs = graphs
        self.topo_feats = topo_feats
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.topo_feats[idx], self.labels[idx]
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

def train(model, loader, optimizer, device, alpha=0.1, task_type="classification"):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    # Loss function for multi-label tasks
    cls_criterion = nn.BCEWithLogitsLoss()

    for batch_graphs, batch_topo, batch_labels in loader:
        batch_graphs = batch_graphs.to(device)
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)
        # Fix shape: [B, 1, num_tasks] -> [B, num_tasks]
        if batch_labels.ndim == 3 and batch_labels.size(1) == 1:
            batch_labels = batch_labels.squeeze(1)

        optimizer.zero_grad()
        logits, proj, _ = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_topo)

        # --- Classification loss ---
        if task_type == "classification":
            # Multi-class: labels are ints [B]
            cls_loss = F.cross_entropy(logits, batch_labels.view(-1).long())
        elif task_type == "binary-classification":
            # Binary but BCE style
            cls_loss = cls_criterion(logits.view(-1), batch_labels.float().view(-1))
        elif task_type == "multi-label":
            # Multi-label: labels are [B, num_tasks]
            #cls_loss = cls_criterion(logits, batch_labels.float())
            mask = ~torch.isnan(batch_labels)
            loss_mat = F.binary_cross_entropy_with_logits(
                logits, batch_labels.float(), reduction="none"
            )
            cls_loss = (loss_mat * mask.float()).sum() / mask.float().sum()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # --- Contrastive loss ---
        g_emb = model.gin_encoder(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
        t_emb = model.topo_encoder(batch_topo)
        con_loss = contrastive_loss(g_emb, t_emb)

        loss = cls_loss + alpha * con_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_graphs.num_graphs

        # Accuracy only for multi-class
        if task_type == "classification":
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_labels.view(-1)).sum().item()
            total_examples += batch_graphs.num_graphs
        else:
            total_examples += batch_graphs.num_graphs

    if task_type == "classification":
        acc = total_correct / total_examples
    else:
        acc = None

    return total_loss / total_examples, acc



@torch.no_grad()
def evaluate(model, loader, device, evaluator=None, task_type="classification"):
    model.eval()
    y_true, y_pred = [], []

    for batch_graphs, batch_topo, batch_labels in loader:
        batch_graphs = batch_graphs.to(device)
        batch_topo = batch_topo.to(device)
        batch_labels = batch_labels.to(device)

        logits, _, _ = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_topo)

        if task_type == "classification":
            preds = logits.argmax(dim=-1)
            y_true.append(batch_labels.view(-1).cpu())
            y_pred.append(preds.cpu())

        elif task_type == "multi-label":
            # store raw logits; evaluator applies sigmoid internally
            y_true.append(batch_labels.view(logits.shape).cpu())
            y_pred.append(logits.cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if task_type == "classification":
        acc = (y_true == y_pred).mean()
        return { "acc": acc }
    elif task_type == "multi-label":
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--c_channels', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-moltoxcast",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    print(dataset[0])
    # Load features and labels
    #dataset = load_data(args.dataset)
    #print(dataset[0])

    list_hks, thres_hks, label = get_thresh_hks(dataset, 10, 0.1)
    list_deg, thres_deg = get_thresh(dataset, 10)
    graph_features = []
    for graph_id in tqdm(range(len(dataset))):
        topo_fe = get_Topo_Fe(dataset[graph_id], list_hks[graph_id], thres_hks)
        # Make sure it's a torch tensor
        topo_fe = torch.tensor(topo_fe, dtype=torch.float)
        graph_features.append(topo_fe)
    topo_tensor = torch.stack(graph_features)
    y = torch.tensor(np.array(label), dtype=torch.float)
    # print(y)

    # Convert to PyTorch tensors
    # Extract dataset details
    num_samples = len(y)
    num_features1 = len(topo_tensor[0])
    num_classes = dataset.num_tasks

    # Define input and output dimensions
    node_feat_dim = dataset[0].x.shape[1]
    topo_dim = num_features1

    split_idx = dataset.get_idx_split()
    X_train, X_val,X_test = topo_tensor[split_idx["train"]], topo_tensor[split_idx["valid"]],topo_tensor[split_idx["test"]]
    y_train, y_val,y_test = y[split_idx["train"]], y[split_idx["valid"]],y[split_idx["test"]]

    data_train = [dataset[i] for i in split_idx["train"]]
    data_valid = [dataset[i] for i in split_idx["valid"]]
    data_test = [dataset[i] for i in split_idx["test"]]

    train_dataset = GraphWithTopoDataset(data_train, X_train, y_train)
    valid_dataset = GraphWithTopoDataset(data_valid, X_val, y_val)
    test_dataset = GraphWithTopoDataset(data_test, X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)


    # Define encoders
    gin_encoder = GINEncoder(in_dim=node_feat_dim, hidden_dim=64, out_dim=128)
    topo_encoder = MLPEncoder(in_dim=topo_dim, hidden_dim=64, out_dim=128)

    # Initialize model, loss function, and optimizer
    # Model
    model = HybridGraphTopoModel(gin_encoder, topo_encoder, hidden_dim=128, proj_dim=64, num_classes=num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss, train_acc = train(model, train_loader, optimizer, device, alpha=0.1, task_type="multi-label")

        print('Evaluating...')
        train_perf = evaluate(model, train_loader, device, evaluator, task_type="multi-label")
        valid_perf = evaluate(model, valid_loader, device, evaluator, task_type="multi-label")
        test_perf = evaluate(model, test_loader, device, evaluator, task_type="multi-label")

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()