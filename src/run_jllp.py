import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.alp import adaptive_label_propagation
from src.model.flp import fixed_alpha_label_propagation
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
    dataset_names = ['gowalla']
    for dataset_name in dataset_names:
        # dataset_name = 'brightkite'
        data, _ = load_dataset(dataset_name)
        num_nodes = data.num_nodes
        print(dataset_name, "valid nodes:", num_nodes, "valid edges:", data.edge_index.shape[1])
        
        # 유사도 계산
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)

        pred = []
        times = []
        trial = 50
        for i in range(trial):
            print(f"JLLP on {dataset_name}: Experiment {i+1} of {trial} Started...")
            start_time = time.time()
            labels = torch.arange(num_nodes)
            pred_lp, _, _ = fixed_alpha_label_propagation(
                data, labels,
                fixed_alpha=0.5,
                structure_similarity=structure_similarity,
                location_similarity=location_similarity
            )
            pred.append(pred_lp)
            save_result(data, pred_lp, dataset_name + '_jllp')
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print(f"JLLP on {dataset_name}: Experiment {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")

        evaluate_and_save_results(
            data, pred,
            dataset_name + "_jllp_result.txt",
            "Label Propagation (Jaccard + Location Similarity):",
            times
        )