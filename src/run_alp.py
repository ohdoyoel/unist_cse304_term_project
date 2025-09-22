import time
import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.alp import adaptive_label_propagation
from src.utils import compute_jaccard_similarity, compute_location_similarity, evaluate_and_save_results, compare_clustering_results, save_result

if __name__ == '__main__':
    dataset_names = ['brightkite', 'gowalla']
    for dataset_name in dataset_names:
        data, _ = load_dataset(dataset_name)
        num_nodes = data.num_nodes
        print(f"Dataset: {dataset_name}, Number of nodes: {num_nodes}, Number of edges: {data.edge_index.shape[1]}")
        print(f"Average degree: {data.avg_degree}")

        # 유사도 계산
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)

        # Ex 1

        # for order in ['original', 'reverse', 'random']:
        #     print(f"ALP on {dataset_name}: Experiment {order} Started...")
        #     pred = []
        #     times = []
        #     iter_info = []
        #     trial = 50
        #     for i in range(trial):
        #         print(f"ALP on {dataset_name}: Experiment {order} {i+1} of {trial} Started...")
        #         start_time = time.time()
        #         labels = torch.arange(num_nodes)
        #         pred_lp, _, iter_info, alpha_info = adaptive_label_propagation(
        #             data, labels,
        #             structure_similarity=structure_similarity,
        #             location_similarity=location_similarity,
        #             verbose=True,
        #             order=order
        #         )
        #         pred.append(pred_lp)
        #         save_result(data, pred_lp, dataset_name + '_alp_' + order)
        #         elapsed_time = time.time() - start_time
        #         times.append(elapsed_time)
        #         print(f"ALP on {dataset_name}: Experiment {order} {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")
        #     print(f"ALP on {dataset_name}: Experiment {order} Ended")

        #     evaluate_and_save_results(
        #         data, pred,
        #         dataset_name + "_alp_result.txt",
        #         "Label Propagation (Adaptive Similarity) " + order + ":",
        #         times,
        #         iter_info
        #     )

        # Ex 2

        pred = []
        times = []
        iters = []
        iter_info = []
        trial = 50
        for i in range(trial):
            print(f"ALP on {dataset_name}: Experiment {i+1} of {trial} Started...")
            start_time = time.time()
            labels = torch.arange(num_nodes)
            pred_lp, _, iter_info, alpha_info = adaptive_label_propagation(
                data, labels,
                structure_similarity=structure_similarity,
                location_similarity=location_similarity,
                verbose=True
            )
            pred.append(pred_lp)
            iters.append(len(alpha_info))
            save_result(data, pred_lp, 'result/'+dataset_name+'/alp', dataset_name + '_alp_' + str(i))
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            print(f"ALP on {dataset_name}: Experiment {i+1} of {trial} Ended in {elapsed_time:.5f} seconds")

        evaluate_and_save_results(
            data, pred,
            dataset_name + "_alp_result.txt",
            "Label Propagation (Adaptive Similarity):",
            times,
            iters,
            result_dir='result/'+dataset_name+'/alp'
        )
        
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