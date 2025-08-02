import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.alp import adaptive_label_propagation
from src.utils import compute_jaccard_similarity, compute_location_similarity, save_graph_result, evaluate_and_save_results

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
    }).drop_duplicates()
    save_graph_result(nodes_df, edges_df, file)

if __name__ == '__main__':
    dataset_name = 'yelp'
    data, _ = load_dataset(dataset_name)
    num_nodes = data.num_nodes
    print(dataset_name, "valid nodes:", num_nodes, "valid edges:", data.edge_index.shape[1])
    labels = torch.arange(num_nodes)
    
    # 유사도 계산
    structure_similarity = compute_jaccard_similarity(data)
    location_similarity = compute_location_similarity(data)

    # 라벨 전파
    start_time = time.time()
    pred_lp, last_adj_dict, iter_info = adaptive_label_propagation(
        data, labels,
        structure_similarity=structure_similarity,
        location_similarity=location_similarity,
        verbose=True
    )
    elapsed_time = time.time() - start_time

    save_result(data, pred_lp, dataset_name + '_alp')

    evaluate_and_save_results(
        data, pred_lp,
        dataset_name + "_alp_result.txt",
        "Label Propagation (Adaptive Similarity):",
        elapsed_time,
        iter_info
    )