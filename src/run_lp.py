import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.lp import label_propagation
from src.utils import evaluate_and_save_results, save_graph_result, compare_clustering_results, save_result

if __name__ == '__main__':
    dataset_names = ['gowalla']
    for dataset_name in dataset_names:
        data, _ = load_dataset(dataset_name)
        num_nodes = data.num_nodes
        print(f"Dataset: {dataset_name}, Number of nodes: {num_nodes}, Number of edges: {data.edge_index.shape[1]}")
        print(f"Average degree: {data.avg_degree}")
        # print(f"Largest WCC nodes: {data.largest_wcc_nodes} ({data.largest_wcc_nodes_fraction}), Largest WCC edges: {data.largest_wcc_edges} ({data.largest_wcc_edges_fraction})")
        # print(f"Largest SCC nodes: {data.largest_scc_nodes} ({data.largest_scc_nodes_fraction}), Largest SCC edges: {data.largest_scc_edges} ({data.largest_scc_edges_fraction})")

        pred = []
        times = []
        iters = []
        trial = 50
        for i in range(trial):
            print(f"LP on {dataset_name}: Experiment {i+1} of {trial} Started...")
            start_time = time.time()
            labels = torch.arange(num_nodes)
            pred_lp, it = label_propagation(data.edge_index, labels)
            pred.append(pred_lp)
            iters.append(it)
            save_result(data, pred_lp, 'result/'+dataset_name+'/lp', dataset_name + '_lp_' + str(i))
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print(f"LP on {dataset_name}: Experiment {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")
        
        evaluate_and_save_results(
            data, 
            pred,
            dataset_name + "_lp_result.txt",  
            "Label Propagation (Majority Voting):",
            times,
            iters,
            result_dir='result/'+dataset_name+'/lp'
        )