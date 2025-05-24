from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import numpy as np
import networkx as nx

def evaluate_clustering(true_labels, pred_labels):
    metrics = {}
    metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    metrics['Homogeneity'] = homogeneity_score(true_labels, pred_labels)
    metrics['Completeness'] = completeness_score(true_labels, pred_labels)
    metrics['V-Measure'] = v_measure_score(true_labels, pred_labels)
    return metrics

def _to_numpy(tensor):
    # Helper to convert torch tensor to numpy array
    return tensor.cpu().numpy()

def _build_graph(edge_index):
    # Helper to build nx.Graph from edge_index torch tensor
    import torch
    G = nx.Graph()
    edges = _to_numpy(edge_index).T
    G.add_edges_from(edges)
    return G

def modularity(edge_index, labels):
    G = _build_graph(edge_index)
    communities = {}
    for idx, label in enumerate(_to_numpy(labels)):
        communities.setdefault(label, []).append(idx)
    comms = list(communities.values())
    return nx.algorithms.community.quality.modularity(G, comms)

def conductance(edge_index, labels):
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    conductances = []
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        if len(nodes) == 0 or len(nodes) == G.number_of_nodes():
            continue
        cut_size = nx.cut_size(G, nodes)
        volume = nx.volume(G, nodes)
        if volume > 0:
            conductances.append(cut_size / volume)
    return float(np.mean(conductances)) if conductances else 0.0

def normalized_cut(edge_index, labels):
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    ncuts = []
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        if len(nodes) == 0 or len(nodes) == G.number_of_nodes():
            continue
        cut = nx.cut_size(G, nodes)
        volA = nx.volume(G, nodes)
        volB = nx.volume(G, list(set(G.nodes) - set(nodes)))
        denom = volA + volB
        if denom > 0:
            ncuts.append(cut / denom)
    return float(np.mean(ncuts)) if ncuts else 0.0

def avg_clustering_coefficient(edge_index, labels):
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    avg_coeffs = []
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        if len(nodes) == 0:
            continue
        subgraph = G.subgraph(nodes)
        avg_coeffs.append(nx.average_clustering(subgraph))
    return float(np.mean(avg_coeffs)) if avg_coeffs else 0.0

def silhouette_coefficient(adj_matrix, labels, sample_size=1000):
    # adj_matrix: torch.Tensor or np.ndarray (n, n)
    # labels: torch.Tensor or np.ndarray (n,)
    from sklearn.metrics import silhouette_score
    import numpy as np

    if hasattr(adj_matrix, 'cpu'):
        # torch.Tensor
        if getattr(adj_matrix, 'is_sparse', False):
            X = adj_matrix.to_dense().cpu().numpy()
        else:
            X = adj_matrix.cpu().numpy()
    else:
        X = adj_matrix
    if hasattr(labels, 'cpu'):
        y = labels.cpu().numpy()
    else:
        y = labels

    n = X.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        X = X[idx][:, idx]
        y = y[idx]

    # Jaccard similarity → distance
    D = 1 - X
    np.fill_diagonal(D, 0)

    try:
        return float(silhouette_score(D, y, metric="precomputed"))
    except Exception:
        return 0.0

def graph_density(edge_index, num_nodes):
    G = _build_graph(edge_index)
    return nx.density(G)