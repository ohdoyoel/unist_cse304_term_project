import numpy as np
import psutil
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import os
import pandas as pd

from src.metrics import avg_clustering_coefficient, conductance, modularity, normalized_cut, silhouette_coefficient

def _to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def _get_edge_pairs(edge_index):
    """Return set of (u, v) pairs from edge_index (numpy array shape [2, N])."""
    edge_array = _to_numpy(edge_index)
    return set(map(tuple, edge_array.T))

def compute_jaccard_similarity(data, edge_index=None):
    """
    Compute Jaccard similarity for connected edges (or given edge_index pairs).
    Returns: dict with (u, v) tuple as key and Jaccard similarity as value.
    """
    if edge_index is None:
        edge_index = data.edge_index
    num_nodes = data.num_nodes

    edge_array = _to_numpy(edge_index)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))

    pairs = set(map(tuple, edge_array.T))
    jaccard = {}
    for u, v in pairs:
        neighbors_u = set(adj[u].indices)
        neighbors_v = set(adj[v].indices)
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        if union > 0:
            jaccard[(u, v)] = intersection / union
    return jaccard

def compute_geometric_similarity(features, edge_index=None):
    """
    Compute cosine similarity for node feature pairs.
    If edge_index is given, only compute for those pairs.
    Returns: dict with (u, v) tuple as key and cosine similarity as value.
    """
    features = _to_numpy(features)
    num_nodes = features.shape[0]
    cos_sim = {}

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-12)

    if edge_index is not None:
        pairs = _get_edge_pairs(edge_index)
        for u, v in pairs:
            sim = np.dot(features_norm[u], features_norm[v])
            if sim > 0:
                cos_sim[(u, v)] = sim
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = np.dot(features_norm[i], features_norm[j])
                if sim > 0:
                    cos_sim[(i, j)] = sim
    return cos_sim

def compute_adaptive_similarity(data, features=None, pred_labels=None):
    """
    Compute adaptive similarity using entropy-based alpha for each node.
    sim_ij = alpha * sim_structure + (1-alpha) * sim_geometry
    Returns: dict with (i, j) tuple as key and adaptive similarity as value, and avg_alpha.
    Only computes for edges present in data.edge_index.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Only compute for edge pairs
    edge_array = _to_numpy(edge_index)
    edge_pairs = set(map(tuple, edge_array.T))

    sim_structure = compute_jaccard_similarity(data)
    if features is None:
        features = getattr(data, 'x', None)
    if features is None:
        raise ValueError("features must be provided or data.x must exist")
    sim_geometry = compute_geometric_similarity(features, edge_index=edge_index)

    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)

    # Build 1-hop neighbor list
    neighbors = defaultdict(list)
    for u, v in edge_pairs:
        neighbors[u].append(v)
        neighbors[v].append(u)

    unique_labels = np.unique(labels[labels != -1])
    n_labels = len(unique_labels)
    log_n_labels = np.log(n_labels) if n_labels > 1 else 1.0

    alpha = np.ones(num_nodes)
    # label index 매핑을 미리 만들어서 속도 개선
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    for i in range(num_nodes):
        neigh = neighbors[i]
        neigh_labels = labels[neigh]
        neigh_labels = neigh_labels[neigh_labels != -1]
        if len(neigh_labels) == 0:
            alpha[i] = 1.0
            continue
        # 빠른 카운팅을 위해 numpy bincount 사용
        idxs = np.array([label_to_idx[l] for l in neigh_labels if l in label_to_idx])
        bincount = np.bincount(idxs, minlength=n_labels)
        p = bincount / len(neigh_labels)
        H = -np.sum(p * np.log(p + 1e-12))
        alpha[i] = 1 - (H / log_n_labels) if log_n_labels > 0 else 1.0
        alpha[i] = np.clip(alpha[i], 0, 1)

    adaptive_sim = {}
    for (i, j) in edge_pairs:
        s = sim_structure.get((i, j), 0.0)
        g = sim_geometry.get((i, j), 0.0)
        a = (alpha[i] + alpha[j]) / 2
        adaptive_sim[(i, j)] = a * s + (1 - a) * g
    avg_alpha = float(np.mean(alpha))
    return adaptive_sim, avg_alpha

def save_graph_result(nodes, edges, filename, result_dir='result'):
    """
    Save nodes and edges DataFrame to CSV files in the result directory.
    """
    os.makedirs(result_dir, exist_ok=True)
    nodes_path = os.path.join(result_dir, filename + "_nodes.csv")
    edges_path = os.path.join(result_dir, filename + "_edges.csv")
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

def evaluate_and_save_results(data, pred_labels, adj_matrix, result_filename, method_name, result_dir='result'):
    """
    Evaluate clustering results and save to a file.
    """
    modularity_score = modularity(data, pred_labels)
    conductance_score = conductance(data, pred_labels)
    avg_clustering_coeff = avg_clustering_coefficient(data, pred_labels)
    normalized_cut_score = normalized_cut(data, pred_labels)
    silhouette_score_value = silhouette_coefficient(adj_matrix, pred_labels)

    print(f"# of Labels: {len(np.unique(pred_labels))}")
    print(f"Modularity: {modularity_score}")
    print(f"Conductance: {conductance_score}")
    print(f"Avg Clustering Coefficient: {avg_clustering_coeff}")
    print(f"Normalized Cut: {normalized_cut_score}")
    print(f"Silhouette Coefficient: {silhouette_score_value}")
    process = psutil.Process(os.getpid())
    print(f"CPU Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    with open(os.path.join(result_dir, result_filename), 'a') as f:
        f.write("\n")
        f.write(f"{method_name}\n")
        f.write(f"# of Labels: {len(np.unique(pred_labels))}\n")
        f.write(f"Modularity: {modularity_score}\n")
        f.write(f"Conductance: {conductance_score}\n")
        f.write(f"Avg Clustering Coefficient: {avg_clustering_coeff}\n")
        f.write(f"Normalized Cut: {normalized_cut_score}\n")
        f.write(f"Silhouette Coefficient: {silhouette_score_value}\n")
        f.write(f"CPU Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB\n")
        if torch.cuda.is_available():
            f.write(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB\n")

def scipy_sparse_to_torch_sparse(sparse_mtx):
    """
    Convert a scipy.sparse CSR/COO matrix to a torch.sparse_coo_tensor.
    """
    if not hasattr(sparse_mtx, 'tocoo'):
        raise ValueError("Input must be a scipy sparse matrix.")
    coo = sparse_mtx.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    i = torch.LongTensor(indices)
    v = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))