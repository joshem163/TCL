# GraphTCL: Cross-View Topology-Aware Graph Representation Learning

This repository contains the official implementation of **GraphTCL**, a dual-view contrastive learning framework for graph classification that aligns **structural graph representations learned by GNNs** with **global topological representations derived from persistent homology**.

📄 **Paper**: *Cross-View Topology-Aware Graph Representation Learning*  
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
- **Backbone-agnostic design** compatible with standard GNNs such as GIN and GCN.
- **Strong empirical performance** on TU benchmarks and OGB molecular datasets.

---

## Architecture

