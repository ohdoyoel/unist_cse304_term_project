import os
import psutil
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.lp import label_propagation
from src.utils import compute_jaccard_similarity, save_graph_result, scipy_sparse_to_torch_sparse

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

def save_result(data, pred_lp, file):
    num_nodes = data.num_nodes
    longitude, latitude = get_long_lat(data, num_nodes)
    # pred_lp가 tuple이면 첫 번째 값만 사용
    if isinstance(pred_lp, tuple):
        pred_lp = pred_lp[0]
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
    save_graph_result(nodes_df, edges_df, file)

if __name__ == '__main__':
    dataset_name = 'brightkite'
    data, _ = load_dataset(dataset_name)
    num_nodes = data.num_nodes

    # 각 노드를 다른 클러스터로 초기화
    labels = torch.arange(num_nodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.randperm(num_nodes)[:int(0.1 * num_nodes)]] = True

    similarity = compute_jaccard_similarity(data)
    from scipy.sparse import lil_matrix
    adj_matrix_sparse = lil_matrix((num_nodes, num_nodes))
    for (u, v), score in similarity.items():
        adj_matrix_sparse[u, v] = score
        adj_matrix_sparse[v, u] = score
    adj_matrix = scipy_sparse_to_torch_sparse(adj_matrix_sparse.tocsr())

    pred_lp = label_propagation(adj_matrix, labels, mask)
    # pred_lp가 tuple이면 첫 번째 값만 사용
    if isinstance(pred_lp, tuple):
        pred_lp = pred_lp[0]
    save_result(data, pred_lp, 'lp')

    from src.utils import evaluate_and_save_results
    evaluate_and_save_results(
        data.edge_index, pred_lp, adj_matrix, 
        "lp_result.txt", 
        "Label Propagation (Jaccard-based):"
    )