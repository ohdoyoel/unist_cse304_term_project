import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.alp import adaptive_label_propagation
from src.model.flp import fixed_alpha_label_propagation
from src.utils import compute_jaccard_similarity, compute_location_similarity, evaluate_and_save_results, save_result

if __name__ == '__main__':
    dataset_names = ['brightkite', 'gowalla']
    for dataset_name in dataset_names:
        data, _ = load_dataset(dataset_name)
        num_nodes = data.num_nodes
        print(dataset_name, "valid nodes:", num_nodes, "valid edges:", data.edge_index.shape[1])
        
        # 유사도 계산
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)

        pred = []
        times = []
        trial = 50
        iters = []
        for i in range(trial):
            print(f"JLLP on {dataset_name}: Experiment {i+1} of {trial} Started...")
            start_time = time.time()
            labels = torch.arange(num_nodes)
            pred_lp, _, it = fixed_alpha_label_propagation(
                data, labels,
                fixed_alpha=0.5,
                structure_similarity=structure_similarity,
                location_similarity=location_similarity,
            )
            pred.append(pred_lp)
            iters.append(it)
            save_result(data, pred_lp, 'result/'+dataset_name+'/jllp', dataset_name + '_jllp_' + str(i))
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print(f"JLLP on {dataset_name}: Experiment {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")

        evaluate_and_save_results(
            data, pred,
            dataset_name + "_jllp_result.txt",
            "Label Propagation (Jaccard + Location Similarity):",
            times,
            iters,
            result_dir='result/'+dataset_name+'/jllp'
        )