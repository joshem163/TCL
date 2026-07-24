import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from sgformer import SGFormer
# try:
#     from torch_geometric.nn import SGFormer
#     HAS_SGFORMER = True
# except Exception:
#     HAS_SGFORMER = False
from torch_geometric.nn import (
    GINConv, SAGEConv, GATConv, GPSConv,GCNConv,
    global_add_pool, global_mean_pool
)
from torch.nn import Linear, Embedding
from torch_geometric.nn import GATv2Conv
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
# class HybridGraphTopoModel(nn.Module):
#     def __init__(self, gin_encoder, topo_encoder, hidden_dim=128, proj_dim=64, num_classes=2):
#         super(HybridGraphTopoModel, self).__init__()
#         self.gin_encoder = gin_encoder
#         self.topo_encoder = topo_encoder
#
#         self.fusion = nn.Linear(
#             self.gin_encoder.out_dim + self.topo_encoder.out_dim, hidden_dim
#         )
#
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#         self.projection_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, proj_dim)
#         )
#
#     def forward(self, x, edge_index, batch, topo_feats):
#         g_emb = self.gin_encoder(x, edge_index, batch)
#         t_emb = self.topo_encoder(topo_feats)
#
#         fused = torch.cat([g_emb, t_emb], dim=-1)
#         fused = F.relu(self.fusion(fused))
#
#         logits = self.classifier(fused)
#         proj = F.normalize(self.projection_head(fused), dim=-1)
#
#         return logits, proj, fused,g_emb, t_emb

import torch
import torch.nn as nn
import torch.nn.functional as F

# class HybridGraphTopoModel(nn.Module):
#     def __init__(self, gin_encoder, topo_encoder, hidden_dim=128, proj_dim=64, num_classes=2, dropout=0.5):
#         super(HybridGraphTopoModel, self).__init__()
#
#         self.gin_encoder = gin_encoder
#         self.topo_encoder = topo_encoder
#         self.dropout = nn.Dropout(dropout)
#
#         self.fusion = nn.Linear(
#             self.gin_encoder.out_dim + self.topo_encoder.out_dim, hidden_dim
#         )
#
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#         self.projection_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, proj_dim)
#         )
#
#     def forward(self, x, edge_index, batch, topo_feats):
#         g_emb = self.gin_encoder(x, edge_index, batch)
#         t_emb = self.topo_encoder(topo_feats)
#
#         fused = torch.cat([g_emb, t_emb], dim=-1)
#         fused = F.relu(self.fusion(fused))
#         fused = self.dropout(fused)
#
#         logits = self.classifier(fused)
#         proj = F.normalize(self.projection_head(fused), dim=-1)
#
#         return logits, proj, fused, g_emb, t_emb

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
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
        return torch.log_softmax(x, dim=-1)

class GINGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_c, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)
class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # Structural encodings (e.g., node degree encoding as Graphormer does)
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
class GCNGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_c, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = x.float()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)
class GraphSAGEGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_c, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = x.float()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)


class GATGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            if i < num_layers - 1:
                self.convs.append(GATConv(in_c, hidden_dim // heads, heads=heads, dropout=dropout))
            else:
                self.convs.append(GATConv(in_c, hidden_dim, heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)


class GPSGraphEncoder(nn.Module):
    """
    Minimal graph-level GPS encoder.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_mpnn = GINEConvWrapper(hidden_dim)
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_mpnn,
                    heads=heads,
                    dropout=dropout,
                    attn_type='multihead'
                )
            )

        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.node_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index, batch=batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)


class GINEConvWrapper(nn.Module):
    """
    Wrapper so GPS can use a local MPNN block.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )

    def forward(self, x, edge_index, **kwargs):
        return self.conv(x, edge_index)


class SGFormerGraphEncoder(nn.Module):
    """
    Uses node-level SGFormer then graph pooling.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        # if not HAS_SGFORMER:
        #     raise ImportError("SGFormer is not available in your current PyG version.")

        self.sgformer = SGFormer(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=num_layers,
            gnn_num_layers=0,
        )
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.sgformer(x, edge_index, batch=batch)
        g = global_mean_pool(x, batch)
        return self.lin_out(g)

class HybridGraphTopoModel(nn.Module):
    def __init__(self, graph_encoder, topo_encoder, hidden_dim, proj_dim, out_dim):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.topo_encoder = topo_encoder

        self.graph_proj = nn.Linear(hidden_dim, proj_dim)
        self.topo_proj = nn.Linear(hidden_dim, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch, topo_feat):
        g_emb = self.graph_encoder(x, edge_index, batch)     # [B, hidden_dim]
        t_emb = self.topo_encoder(topo_feat)                 # [B, hidden_dim]

        g_proj = self.graph_proj(g_emb)
        t_proj = self.topo_proj(t_emb)

        fused = torch.cat([g_emb, t_emb], dim=-1)
        logits = self.classifier(fused)
        return logits, g_proj, t_proj