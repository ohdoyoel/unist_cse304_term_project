import os
import psutil
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.lp import label_propagation
from src.utils import compute_adaptive_similarity, save_graph_result, scipy_sparse_to_torch_sparse
from src.metrics import modularity, conductance, normalized_cut, avg_clustering_coefficient, silhouette_coefficient, graph_density

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"CPU Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

def get_long_lat(data, num_nodes):
    longitude = (
        data.longitude.cpu().numpy() if hasattr(data, 'longitude') and hasattr(data.longitude, 'cpu')
        else data.longitude if hasattr(data, 'longitude')
        else np.full(num_nodes, np.nan)
    )
    latitude = (
        data.latitude.cpu().numpy() if hasattr(data, 'latitude') and hasattr(data.latitude, 'cpu')
        else data.latitude if hasattr(data, 'latitude')
        else np.full(num_nodes, np.nan)
    )
    return longitude, latitude

def save_result(data, pred_lp, nodes_filename, edges_filename):
    num_nodes = data.num_nodes
    longitude, latitude = get_long_lat(data, num_nodes)
    nodes_df = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'cluster_label': pred_lp.cpu().numpy(),
        'longitude': longitude,
        'latitude': latitude
    })
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    edges_df = pd.DataFrame({
        'source': edge_array[0],
        'target': edge_array[1]
    }).T.drop_duplicates().T
    save_graph_result(nodes_df, edges_df, nodes_filename, edges_filename)

if __name__ == '__main__':
    data, _ = load_dataset('brightkite')
    num_nodes = data.num_nodes
    k = 3
    labels = torch.randint(0, k, (num_nodes,))
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.randperm(num_nodes)[:int(0.1 * num_nodes)]] = True

    similarity = compute_adaptive_similarity(data, features=data.x)
    from scipy.sparse import lil_matrix
    adj_matrix_sparse = lil_matrix((num_nodes, num_nodes))
    for (u, v), score in similarity.items():
        adj_matrix_sparse[u, v] = score
        adj_matrix_sparse[v, u] = score
    adj_matrix = scipy_sparse_to_torch_sparse(adj_matrix_sparse.tocsr())

    pred_lp = label_propagation(adj_matrix, labels, mask)
    save_result(data, pred_lp, 'adaptive_nodes.csv', 'adaptive_edges.csv')

    print("Label Propagation (Adaptive Similarity):")
    print("Modularity:", modularity(data.edge_index, pred_lp))
    print("Conductance:", conductance(data.edge_index, pred_lp))
    print("Normalized Cut:", normalized_cut(data.edge_index, pred_lp))
    print("Avg Clustering Coefficient:", avg_clustering_coefficient(data.edge_index, pred_lp))
    print("Silhouette Coefficient:", silhouette_coefficient(adj_matrix, pred_lp, sample_size=1000))
    print("Graph Density:", graph_density(data.edge_index, data.num_nodes))

    print_memory_usage()