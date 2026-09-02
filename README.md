# GraphTCL: Topology-Aware Contrastive Learning for Graph Classificatio

This repository contains the official implementation of **GraphTCL**, a dual-view contrastive learning framework for graph classification that aligns **structural graph representations learned by GNNs** with **global topological representations derived from persistent homology**.

📄 **Paper**: *GraphTCL: Topology-Aware Contrastive Learning for Graph Classificatio*  
📌 **arXiv**: Link will be added later

---

## Overview

Graph Neural Networks (GNNs) are effective at capturing local structural patterns via message passing, but they often fail to represent **global topological structure** such as cycles and higher-order connectivity.  

**GraphTCL** addresses this limitation by introducing a **dual-view learning framework**:

- **Structural View**: Learned using a message-passing GNN (e.g., GIN or GCN).
- **Topological View**: Extracted using **persistent homology (PH)** computed from graph filtrations based on the **Heat Kernel Signature (HKS)**.
- **Cross-View Contrastive Learning**: A bidirectional contrastive loss explicitly aligns structural and topological embeddings in a shared latent space.

The aligned representations are fused and used for supervised graph classification.

---

## Key Contributions

- **Dual-view contrastive framework** aligning structural (GNN) and topological (PH) representations.
- **Topology-aware representation learning** treating topology as an independent modality rather than an auxiliary feature.
- **Backbone-flixible design** compatible with standard GNNs such as GIN and GCN.
- **Strong empirical performance** on TU benchmarks and OGB molecular datasets.




## Architecture
![GraphTCL](https://github.com/user-attachments/assets/c1cfac2f-a2ae-4dcb-9a14-246aa57d2265)


---

## Datasets

### TU Benchmark Datasets
- MUTAG  
- BZR  
- PTC  
- COX2  
- PROTEINS  
- IMDB-BINARY  
- IMDB-MULTI  

### OGB Molecular Datasets
- ogbg-molbace  
- ogbg-molclintox  
- ogbg-molbbbp  
- ogbg-molhiv  
- ogbg-molsider
- ogbg-moltoxcast 
- ogbg-moltox21

---

## Implementation Details

- **Framework**: PyTorch, PyTorch Geometric  
- **GNN Backbones**: GIN, GCN  
- **Topological Features**:
  - Persistent Homology
  - Heat Kernel Signature (HKS) filtration
  - Vectorizations (e.g. Betti curves)
- **Training Protocol**:
  - TU datasets: 10-fold cross-validation
  - OGB datasets: official scaffold splits
- **Loss Function**:
  \[
  \mathcal{L} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{con}
  \]
  with \(\alpha = 0.1\)

---

# Train GraphTCL on a TU dataset
python graphtcl_train_accuracy_grid_search.py \
  --dataset MUTAG \
  --gnn gin \
  --topo hks \
  --alpha 0.1


