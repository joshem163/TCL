import torch
from torch_geometric.utils import degree
import statistics
import pandas as pd
import numpy as np
import torch
import networkx as nx
import pyflagser
from collections import Counter


def process_thresholds(lst, N):
    if N < 2:
        raise ValueError("N must be at least 2 to include min and max thresholds.")

    # Count occurrences of each value
    count = Counter(lst)

    # Find the minimum and maximum values
    min_val, max_val = min(lst), max(lst)

    # Remove min and max from the counting
    count.pop(min_val, None)
    count.pop(max_val, None)

    # Select the N-1 values with the highest counts
    top_values = sorted(count.items(), key=lambda x: x[1], reverse=True)[:N - 1]

    # Prepare the thresholds: a_0=min, top N-1 values, a_N=max
    thresholds = [min_val] + [value for value, _ in top_values] + [max_val]

    return sorted(thresholds)
def get_degree_centrality(data):
    # Assume edge_index is of shape [2, num_edges]
    # and contains edges in COO format

    edge_index = data.edge_index  # shape: [2, E]
    num_nodes = data.num_nodes    # or x.shape[0]

    # Compute degree for each node
    deg = degree(edge_index[0], num_nodes=num_nodes)  # degree of each node

    # Normalize to get degree centrality (divide by max possible degree)
    deg_centrality = deg / (num_nodes - 1)
    return deg_centrality

def get_atomic_weight(data):
    # Assume edge_index is of shape [2, num_edges]
    # and contains edges in COO format
    Atomic_weight=[data.x[i][0] for i in range(len(data.x))]
    return torch.tensor(Atomic_weight)

from torch_geometric.utils import to_networkx


def compute_hks(graph, t_values):
    """
    Compute the Heat Kernel Signature (HKS) for each node in the graph.
    :param graph: NetworkX graph (undirected, unweighted)
    :param t_values: List of diffusion time values to compute HKS
    :return: Dictionary with nodes as keys and HKS values as lists
    """
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)

    hks = []

    heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t_values * eigvals)), eigvecs.T))
    for i, node in enumerate(graph.nodes()):
        hks.append(heat_kernel[i, i])
    return torch.tensor(hks)

def get_thres_atom(dataset,number_threshold):
    thresh = []
    graph_list = []

    for graph_id in range(len(dataset)):
        atomic_values=get_atomic_weight(dataset[graph_id])
        graph_list.append(atomic_values)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh)
def get_thresh(dataset,number_threshold):
    thresh = []
    graph_list = []

    for graph_id in range(len(dataset)):
        degree_centrality_values=get_degree_centrality(dataset[graph_id])
        graph_list.append(degree_centrality_values)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh)
def get_thresh_hks(dataset,number_threshold,t_value):
    graph_list = []
    label=[]
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        hks_values=compute_hks(graph,t_value)
        graph_list.append(hks_values)
        label.append(dataset[graph_id].y)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh),label


def get_Topo_Fe(graph, feature, threshold):
    """
    Compute Betti-0, Betti-1, num_nodes, num_edges for combinations of thresholds.
    All inputs are torch tensors.
    """
    betti0_row = []
    betti1_row = []
    nodes_row = []
    edges_row = []

    # edge_index is [2, num_edges]
    edge_index = graph.edge_index  # already a torch tensor

    # Ensure 1D for features and thresholds
    feature = feature.view(-1)

    # Convert thresholds to 1D tensor if needed
    threshold = threshold.view(-1)

    num_nodes_total = feature.shape[0]

    for p in range(threshold.size(0)):
            # Find active nodes as intersection
        idx1 = torch.where(feature <= threshold[p])[0]
        n_active = torch.tensor(list(set(idx1.tolist())), dtype=torch.long)

        if n_active.numel() == 0:
            betti0_row.append(0)
            betti1_row.append(0)
            nodes_row.append(0)
            edges_row.append(0)
        else:
            # Create graph using only active nodes
            active_set = set(n_active.tolist())
            G = nx.Graph()
            G.add_nodes_from(active_set)

            # Add edges where both endpoints are active
            u = edge_index[0].tolist()
            v = edge_index[1].tolist()
            for uu, vv in zip(u, v):
                if uu in active_set and vv in active_set:
                    G.add_edge(int(uu), int(vv))

            # Compute Betti numbers
            Adj = nx.to_numpy_array(G, nodelist=sorted(active_set))
            my_flag = pyflagser.flagser_unweighted(
                Adj, min_dimension=0, max_dimension=2,
                directed=False, coeff=2, approximation=None
            )
            x = my_flag["betti"]

            betti0_row.append(int(x[0]))
            betti1_row.append(int(x[1]) if len(x) > 1 else 0)
            nodes_row.append(len(active_set))
            edges_row.append(G.number_of_edges())

    # Convert results to torch tensors for consistency
    return torch.cat((torch.tensor(betti0_row, dtype=torch.float),
            torch.tensor(betti1_row, dtype=torch.float),
            torch.tensor(nodes_row, dtype=torch.float),
            torch.tensor(edges_row, dtype=torch.float)),dim=0)



def Topo_Fe_TimeSeries_MP(graph, feature1, feature2, threshold1, threshold2):
    """
    Compute Betti-0, Betti-1, num_nodes, num_edges for combinations of thresholds.
    All inputs are torch tensors.
    """
    betti_0_all = []
    betti_1_all = []
    num_nodes_all = []
    num_edges_all = []

    # edge_index is [2, num_edges]
    edge_index = graph.edge_index  # already a torch tensor

    # Ensure 1D for features and thresholds
    feature1 = feature1.view(-1)
    feature2 = feature2.view(-1)

    # Convert thresholds to 1D tensor if needed
    threshold1 = threshold1.view(-1)
    threshold2 = threshold2.view(-1)

    num_nodes_total = feature1.shape[0]

    for p in range(threshold1.size(0)):
        betti0_row = []
        betti1_row = []
        nodes_row = []
        edges_row = []

        for q in range(threshold2.size(0)):
            # Find active nodes as intersection
            idx1 = torch.where(feature1 <= threshold1[p])[0]
            idx2 = torch.where(feature2 <= threshold2[q])[0]
            n_active = torch.tensor(list(set(idx1.tolist()) & set(idx2.tolist())), dtype=torch.long)

            if n_active.numel() == 0:
                betti0_row.append(0)
                betti1_row.append(0)
                nodes_row.append(0)
                edges_row.append(0)
            else:
                # Create graph using only active nodes
                active_set = set(n_active.tolist())
                G = nx.Graph()
                G.add_nodes_from(active_set)

                # Add edges where both endpoints are active
                u = edge_index[0].tolist()
                v = edge_index[1].tolist()
                for uu, vv in zip(u, v):
                    if uu in active_set and vv in active_set:
                        G.add_edge(int(uu), int(vv))

                # Compute Betti numbers
                Adj = nx.to_numpy_array(G, nodelist=sorted(active_set))
                my_flag = pyflagser.flagser_unweighted(
                    Adj, min_dimension=0, max_dimension=2,
                    directed=False, coeff=2, approximation=None
                )
                x = my_flag["betti"]

                betti0_row.append(int(x[0]))
                betti1_row.append(int(x[1]) if len(x) > 1 else 0)
                nodes_row.append(len(active_set))
                edges_row.append(G.number_of_edges())

        betti_0_all.append(betti0_row)
        betti_1_all.append(betti1_row)
        num_nodes_all.append(nodes_row)
        num_edges_all.append(edges_row)

    # Convert results to torch tensors for consistency
    return (torch.tensor(betti_0_all, dtype=torch.float),
            torch.tensor(betti_1_all, dtype=torch.float),
            torch.tensor(num_nodes_all, dtype=torch.float),
            torch.tensor(num_edges_all, dtype=torch.float))

def stat(acc_list, metric):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final', metric, f'using 5 fold CV: {mean:.4f} \u00B1 {stdev:.4f}%')


def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_ac = np.max(train_acc)
    test_ac = np.max(test_acc)
    #print(f'Train accuracy = {train_ac:.4f}%,Test Accuracy = {test_ac:.4f}%\n')
    return test_ac, best_result
def apply_Zscore(MP_tensor):
# MP_tensor: (N, 4, 10, 10)
    # Compute mean and std per channel (dim=0 = graphs, dim=2,3 = grid)
    mean = MP_tensor.mean(dim=(0, 2, 3), keepdim=True)   # shape (1,4,1,1)
    std  = MP_tensor.std(dim=(0, 2, 3), keepdim=True)    # shape (1,4,1,1)

    # Apply z-score normalization per channel
    MP_tensor_z = (MP_tensor - mean) / (std + 1e-8)  # avoid div by zero
    return MP_tensor_z