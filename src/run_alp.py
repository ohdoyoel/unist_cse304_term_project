import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.alp import adaptive_label_propagation
from src.utils import compute_jaccard_similarity, compute_location_similarity, save_graph_result, evaluate_and_save_results, compare_clustering_results

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
    dataset_names = ['yelp', 'brightkite', 'gowalla']
    # dataset_names = ['custom']
    for dataset_name in dataset_names:
        # dataset_name = 'brightkite'
        data, _ = load_dataset(dataset_name)
        num_nodes = data.num_nodes
        print(f"Dataset: {dataset_name}, Number of nodes: {num_nodes}, Number of edges: {data.edge_index.shape[1]}")
        print(f"Average degree: {data.avg_degree}")

        # 유사도 계산
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)

        # Ex 1

        for order in ['original', 'reverse', 'random']:
            print(f"ALP on {dataset_name}: Experiment {order} Started...")
            pred = []
            times = []
            iter_info = []
            trial = 50
            for i in range(trial):
                print(f"ALP on {dataset_name}: Experiment {order} {i+1} of {trial} Started...")
                start_time = time.time()
                labels = torch.arange(num_nodes)
                pred_lp, _, iter_info, alpha_info = adaptive_label_propagation(
                    data, labels,
                    structure_similarity=structure_similarity,
                    location_similarity=location_similarity,
                    verbose=True,
                    order=order
                )
                pred.append(pred_lp)
                save_result(data, pred_lp, dataset_name + '_alp_' + order)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                print(f"ALP on {dataset_name}: Experiment {order} {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")
            print(f"ALP on {dataset_name}: Experiment {order} Ended")

            evaluate_and_save_results(
                data, pred,
                dataset_name + "_alp_result.txt",
                "Label Propagation (Adaptive Similarity) " + order + ":",
                times,
                iter_info
            )

        # Ex 2

        # pred = []
        # times = []
        # iter_info = []
        # trial = 50
        # for i in range(trial):
        #     print(f"ALP on {dataset_name}: Experiment {i+1} of {trial} Started...")
        #     start_time = time.time()
        #     labels = torch.arange(num_nodes)
        #     pred_lp, _, iter_info, alpha_info = adaptive_label_propagation(
        #         data, labels,
        #         structure_similarity=structure_similarity,
        #         location_similarity=location_similarity,
        #         verbose=True
        #     )
        #     pred.append(pred_lp)
        #     save_result(data, pred_lp, dataset_name + '_alp')
        #     elapsed_time = time.time() - start_time
        #     times.append(elapsed_time)
        #     print(f"ALP on {dataset_name}: Experiment {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")

        # evaluate_and_save_results(
        #     data, pred,
        #     dataset_name + "_alp_result.txt",
        #     "Label Propagation (Adaptive Similarity):",
        #     times,
        #     iter_info
        # )
        
        # alpha 값들을 CSV 파일로 저장
        # alpha_df = pd.DataFrame(alpha_info)
        # alpha_df.to_csv(f"result/{dataset_name}_alpha.csv", index=False)

        # Ex 3 : Custom Dataset

        # print(f"ALP on {dataset_name}: Experiment Started...")
        # start_time = time.time()
        # labels = torch.arange(num_nodes)
        # pred_lp, _, iter_info, alpha_info = adaptive_label_propagation(
        #     data, labels,
        #     structure_similarity=structure_similarity,
        #     location_similarity=location_similarity,
        #     verbose=True,
        #     save_plot=True
        # )
        # print(f"ALP on {dataset_name}: Experiment Ended")
        # print(pred_lp)