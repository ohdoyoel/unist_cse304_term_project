import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.lv import louvain_method
from src.utils import evaluate_and_save_results, save_graph_result, scipy_sparse_to_torch_sparse

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

def save_result(data, community_labels, file):
    num_nodes = data.num_nodes
    longitude, latitude = get_long_lat(data, num_nodes)
    nodes_df = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'cluster_labels': community_labels,
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

    from scipy.sparse import coo_matrix
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    adj_matrix_sparse = coo_matrix(
        (np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
        shape=(num_nodes, num_nodes)
    )
    adj_matrix = scipy_sparse_to_torch_sparse(adj_matrix_sparse)

    community_labels = louvain_method(adj_matrix)
    save_result(data, community_labels, 'lv')

    from src.utils import evaluate_and_save_results
    import torch
    evaluate_and_save_results(
        data.edge_index, torch.tensor(community_labels), adj_matrix,
        "lv_result.txt",
        "Louvain Method:"
    )