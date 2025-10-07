import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128, num_layers=3):
        super(GINEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn1))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            nnk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nnk))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Final projection
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Graph-level embedding
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        return self.fc_out(x)
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128, num_layers=2):
        super(MLPEncoder, self).__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(dims[i+1]))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.mlp(x)
class HybridGraphTopoModel(nn.Module):
    def __init__(self, gin_encoder, topo_encoder, hidden_dim=128, proj_dim=64, num_classes=2):
        super(HybridGraphTopoModel, self).__init__()
        self.gin_encoder = gin_encoder
        self.topo_encoder = topo_encoder

        self.fusion = nn.Linear(
            self.gin_encoder.out_dim + self.topo_encoder.out_dim, hidden_dim
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, edge_index, batch, topo_feats):
        g_emb = self.gin_encoder(x, edge_index, batch)
        t_emb = self.topo_encoder(topo_feats)

        fused = torch.cat([g_emb, t_emb], dim=-1)
        fused = F.relu(self.fusion(fused))

        logits = self.classifier(fused)
        proj = F.normalize(self.projection_head(fused), dim=-1)

        return logits, proj, fused
