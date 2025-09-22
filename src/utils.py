import numpy as np
import psutil
from sklearn.metrics import adjusted_rand_score
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import os
import pandas as pd

from src.metrics import avg_clustering_coefficient, conductance, density_score, intra_cluster_avg_distance, inter_cluster_avg_distance, modularity, normalized_cut, sn_modularity, spatial_silhouette

def _to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def _get_edge_pairs(edge_index):
    """Return set of (u, v) pairs from edge_index (numpy array shape [2, N])."""
    edge_array = _to_numpy(edge_index)
    return set(map(tuple, edge_array.T))

# 0.1 단위로 도수분포 계산 및 출력
def print_distribution(values, bins=10, name=None):
    # 최소값과 최대값을 반올림하여 구간 설정
    min_val = np.floor(np.min(values) * 10) / 10
    max_val = np.ceil(np.max(values) * 10) / 10
    bins = np.linspace(min_val, max_val, 11)  # 10개 구간으로 균등 분할
    
    print(f"{name} 분포:")

    hist, bins = np.histogram(values, bins=bins)
    total = len(values)
    if name == "클러스터별 노드 수":
        # 첫번째 bin을 10개 단위로 세분화
        first_bin_max = bins[1]
        first_bin_values = values[values < first_bin_max]
        if len(first_bin_values) > 0:
            sub_bins = np.arange(bins[0], bins[1], 10)  # 10개 단위로 나누기
            sub_bins = np.append(sub_bins, bins[1])  # 마지막 경계 추가
            sub_hist, sub_bins = np.histogram(first_bin_values, bins=sub_bins)
            for i in range(len(sub_hist)):
                start, end = sub_bins[i], sub_bins[i+1]
                count = sub_hist[i]
                percentage = (count / total) * 100
                print(f"  {start:.1f}-{end:.1f}: {count:5d} ({percentage:5.1f}%)")
        
    for i in range(len(hist)):
        start, end = bins[i], bins[i+1]
        count = hist[i]
        percentage = (count / total) * 100
        print(f"  {start:.1f}-{end:.1f}: {count:5d} ({percentage:5.1f}%)")
    print(f"  평균: {np.mean(values):.5f}, 표준편차: {np.std(values):.5f}")
    print(f"  최소: {np.min(values):.5f}, 최대: {np.max(values):.5f}")

def compute_jaccard_similarity(data, edge_index=None):
    """
    Compute Jaccard similarity for connected edges (or given edge_index pairs).
    Returns: dict with (u, v) tuple as key and Jaccard similarity as value.
    """
    print("Computing Jaccard similarity...")
    if edge_index is None:
        edge_index = data.edge_index
    num_nodes = data.num_nodes

    edge_array = _to_numpy(edge_index)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))

    pairs = set(map(tuple, edge_array.T))
    jaccard = {}
    for u, v in pairs:
        neighbors_u = set(adj[u].indices)
        neighbors_u.add(u)
        neighbors_v = set(adj[v].indices)
        neighbors_v.add(v)
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        if union > 0:
            jaccard[(u, v)] = intersection / union
    
    # 유사도 z-score 정규화
    # mean = np.mean(list(jaccard.values()))
    # std = np.std(list(jaccard.values()))
    # jaccard = {k: (v - mean) / std for k, v in jaccard.items()}

    print("Jaccard similarity computation completed")
    print_distribution(list(jaccard.values()), name="Jaccard similarity")

    return jaccard

def compute_location_similarity(data, edge_index=None):
    """
    Compute location similarity for connected edges (or given edge_index pairs) using haversine distance.
    Returns: dict with (u, v) tuple as key and location similarity as value.
    """
    print("Computing location similarity...")
    if edge_index is None:
        edge_index = data.edge_index
    
    edge_array = _to_numpy(edge_index)
    coords = _to_numpy(data.rad_x)
    
    # Haversine 거리 계산 함수
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # 지구의 반지름 (km)
        
        # Haversine 공식
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # # 거리를 유사도로 변환 (거리가 멀수록 유사도는 감소)
        # # 지수 감소 함수를 사용하여 거리가 0일 때 1, 멀어질수록 0에 가까워지게 함
        # similarity = np.exp(-distance/1000)
        return distance
    
    # 각 엣지에 대해 유사도 계산
    distances = {}
    for i, j in zip(edge_array[0], edge_array[1]):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        dis = haversine_distance(lat1, lon1, lat2, lon2)
        distances[(i, j)] = dis
        distances[(j, i)] = dis  # 무방향 그래프이므로 양방향 모두 저장

    # 유사도 계산 (거리가 멀수록 유사도는 감소)
    max_dist = max(distances.values())
    similarities = {k: 1 - v/max_dist for k, v in distances.items()}
    
    # 유사도 z-score 정규화
    # mean = np.mean(list(similarities.values()))
    # std = np.std(list(similarities.values()))
    # similarities = {k: (v - mean) / std for k, v in similarities.items()}
    
    print("Location similarity computation completed")
    print_distribution(list(similarities.values()), name="Location similarity")

    return similarities

def compute_geometric_similarity(features, edge_index=None):
    """
    Compute cosine similarity for node feature pairs.
    If edge_index is given, only compute for those pairs.
    Returns: dict with (u, v) tuple as key and cosine similarity as value.
    """
    features = _to_numpy(features)
    num_nodes = features.shape[0]
    cos_sim = {}

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-12)

    if edge_index is not None:
        pairs = _get_edge_pairs(edge_index)
        for u, v in pairs:
            sim = np.dot(features_norm[u], features_norm[v])
            if sim > 0:
                cos_sim[(u, v)] = sim
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = np.dot(features_norm[i], features_norm[j])
                if sim > 0:
                    cos_sim[(i, j)] = sim
    return cos_sim


def compute_adaptive_similarity(data, structure_similarity, location_similarity, pred_labels=None):
    """
    적응형 유사도 계산 함수
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        structure_similarity: 미리 계산된 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        location_similarity: 미리 계산된 위치 기반 유사도 딕셔너리 {(u,v): similarity_score}
        pred_labels: 현재 예측된 레이블 (기본값: None, data.y 사용)
    
    Returns:
        adaptive_sim: 적응형 유사도 딕셔너리 {(u,v): similarity_score}
        avg_alpha: 평균 알파 값
        dev_alpha: 알파 값의 표준편차
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_array = _to_numpy(edge_index)
    
    # 무방향 그래프를 위한 대칭 행렬 생성 (CSR 형식 사용)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    # labels 준비
    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)
    
    # 유효한 레이블만 선택
    valid_mask = (labels != -1) & ~np.isnan(labels)
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # 모든 노드에 대해 알파값 계산 (수렴 문제 해결을 위해)
    nodes_to_compute = np.arange(num_nodes)
    
    # 노드별 알파값 계산
    alpha = np.full(num_nodes, 0.5)

    for node in nodes_to_compute:
        neighbors = adj[node].indices
        if len(neighbors) == 0:
            alpha[node] = 0
            continue
            
        neigh_labels = labels[neighbors]
        valid_neigh = neigh_labels[neigh_labels != -1]
            
        # 엔트로피 계산
        label_counts = np.bincount([label_to_idx[l] for l in valid_neigh])
        label_counts = label_counts[label_counts > 0]  # 0이 아닌 레이블 카운트만 사용
        
        # 최대 엔트로피 계산 (모든 레이블이 균등하게 분포된 경우)
        n_unique_labels = len(label_counts)
        max_entropy = np.log(n_unique_labels) if n_unique_labels > 1 else 1.0
        # max_entropy = np.log(len(unique_labels)) if len(unique_labels) > 1 else 1.0 # 비교를 위해 모든 레이블 수 사용
        
        # 실제 엔트로피 계산
        probs = label_counts / len(valid_neigh)
        entropy = -np.sum(probs * np.log(probs))
        
        # 정규화된 엔트로피 (0~1 사이 값)
        normalized_entropy = entropy / max_entropy if max_entropy != 0 else 0
        normalized_entropy = np.clip(normalized_entropy, 0, 1)
        
        # 알파값 계산: 노드의 구조적 신뢰도를 나타냄
        # - 이웃들이 모두 같은 레이블이면 alpha=1 (구조가 매우 명확함)
        # - 이웃들의 레이블이 다양할수록 alpha=0에 가까워짐 (구조가 모호함)
        # 이 값은 해당 노드가 다른 노드에게 구조적 정보를 전달할 때의 신뢰도로 사용됨
        alpha[node] = 1 - normalized_entropy
    
    # 적응형 유사도 계산
    adaptive_sim = {}
    edges_to_compute = np.arange(edge_array.shape[1])  # 모든 엣지에 대해 계산
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # 미리 계산된 유사도 사용
        sim_structure = structure_similarity[(i, j)]
        sim_location = location_similarity[(i, j)]

        # 적응형 가중치 적용
        # alpha[j]를 사용: j(송신자)의 구조적 신뢰도에 따라 구조적 유사도의 영향력 조절
        # - j의 이웃이 일관된 레이블을 가질 때(alpha[j]가 높을 때) → 구조적 유사도를 더 신뢰
        # - j의 이웃이 다양한 레이블을 가질 때(alpha[j]가 낮을 때) → 위치 유사도에 더 의존
        adaptive_sim[(i, j)] = alpha[j] * sim_structure + (1 - alpha[j]) * sim_location
        
        # a = (alpha[i] + alpha[j]) / 2
        # adaptive_sim[(i, j)] = a * sim_structure + (1 - a) * sim_location
    
    # 통계 계산 및 디버깅을 위한 분포 출력
    computed_alphas = alpha  # 모든 노드의 알파값 사용
    avg_alpha = float(np.mean(computed_alphas))
    dev_alpha = float(np.std(computed_alphas))
    
    # 알파값 분포
    # print_distribution(computed_alphas, name="알파값")
    
    # 적응형 유사도 분포
    # adaptive_sims = np.array([sim for sim in adaptive_sim.values()])
    # print_distribution(adaptive_sims, name="적응형 유사도")
    
    return adaptive_sim, alpha, avg_alpha, dev_alpha

def compute_adaptive_similarity_without_location_similarity(data, structure_similarity, pred_labels=None):
    """
    적응형 유사도 계산 함수
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        structure_similarity: 미리 계산된 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        pred_labels: 현재 예측된 레이블 (기본값: None, data.y 사용)
    
    Returns:
        adaptive_sim: 적응형 유사도 딕셔너리 {(u,v): similarity_score}
        avg_alpha: 평균 알파 값
        dev_alpha: 알파 값의 표준편차
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_array = _to_numpy(edge_index)
    
    # 무방향 그래프를 위한 대칭 행렬 생성 (CSR 형식 사용)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    # labels 준비
    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)
    
    # 유효한 레이블만 선택
    valid_mask = (labels != -1) & ~np.isnan(labels)
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # 모든 노드에 대해 알파값 계산 (수렴 문제 해결을 위해)
    nodes_to_compute = np.arange(num_nodes)
    
    # 노드별 알파값 계산
    alpha = np.full(num_nodes, 0.5)

    for node in nodes_to_compute:
        neighbors = adj[node].indices
        if len(neighbors) == 0:
            continue
            
        neigh_labels = labels[neighbors]
        valid_neigh = neigh_labels[neigh_labels != -1]
        if len(valid_neigh) == 0:
            continue
            
        # 엔트로피 계산
        label_counts = np.bincount([label_to_idx[l] for l in valid_neigh])
        label_counts = label_counts[label_counts > 0]  # 0이 아닌 레이블 카운트만 사용
        
        # 최대 엔트로피 계산 (모든 레이블이 균등하게 분포된 경우)
        n_unique_labels = len(label_counts)
        max_entropy = np.log(n_unique_labels) if n_unique_labels > 1 else 1.0
        # max_entropy = np.log(len(unique_labels)) if len(unique_labels) > 1 else 1.0 # 비교를 위해 모든 레이블 수 사용
        
        # 실제 엔트로피 계산
        probs = label_counts / len(valid_neigh)
        entropy = -np.sum(probs * np.log(probs))
        
        # 정규화된 엔트로피 (0~1 사이 값)
        normalized_entropy = entropy / max_entropy if max_entropy != 0 else 0
        normalized_entropy = np.clip(normalized_entropy, 0, 1)
        
        # 알파값 계산: 노드의 구조적 신뢰도를 나타냄
        # - 이웃들이 모두 같은 레이블이면 alpha=1 (구조가 매우 명확함)
        # - 이웃들의 레이블이 다양할수록 alpha=0에 가까워짐 (구조가 모호함)
        # 이 값은 해당 노드가 다른 노드에게 구조적 정보를 전달할 때의 신뢰도로 사용됨
        alpha[node] = 1 - normalized_entropy
    
    # 적응형 유사도 계산
    adaptive_sim = {}
    edges_to_compute = np.arange(edge_array.shape[1])  # 모든 엣지에 대해 계산
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # 미리 계산된 유사도 사용
        sim_structure = structure_similarity[(i, j)]

        # 적응형 가중치 적용
        # alpha[j]를 사용: j(송신자)의 구조적 신뢰도에 따라 구조적 유사도의 영향력 조절
        # - j의 이웃이 일관된 레이블을 가질 때(alpha[j]가 높을 때) → 구조적 유사도를 더 신뢰
        # - j의 이웃이 다양한 레이블을 가질 때(alpha[j]가 낮을 때) → 위치 유사도에 더 의존
        adaptive_sim[(i, j)] = alpha[j] * sim_structure
    
    # 통계 계산 및 디버깅을 위한 분포 출력
    computed_alphas = alpha  # 모든 노드의 알파값 사용
    avg_alpha = float(np.mean(computed_alphas))
    dev_alpha = float(np.std(computed_alphas))
    
    # 알파값 분포
    print_distribution(computed_alphas, name="알파값")
    
    # 적응형 유사도 분포
    adaptive_sims = np.array([sim for sim in adaptive_sim.values()])
    print_distribution(adaptive_sims, name="적응형 유사도")
    
    return adaptive_sim, avg_alpha, dev_alpha

def compute_fixed_alpha_similarity(data, fixed_alpha, structure_similarity, location_similarity, pred_labels=None):
    """
    적응형 유사도 계산 함수
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        structure_similarity: 미리 계산된 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        location_similarity: 미리 계산된 위치 기반 유사도 딕셔너리 {(u,v): similarity_score}
        pred_labels: 현재 예측된 레이블 (기본값: None, data.y 사용)
    
    Returns:
        adaptive_sim: 적응형 유사도 딕셔너리 {(u,v): similarity_score}
        avg_alpha: 평균 알파 값
        dev_alpha: 알파 값의 표준편차
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_array = _to_numpy(edge_index)
    
    # 무방향 그래프를 위한 대칭 행렬 생성 (CSR 형식 사용)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    # labels 준비
    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)
    
    # 유효한 레이블만 선택
    valid_mask = (labels != -1) & ~np.isnan(labels)
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # 전파가 필요한 노드 식별 (레이블이 다른 엣지의 노드들)
    nodes_to_compute = set()
    diff_labels = labels[edge_array[0]] != labels[edge_array[1]]
    if diff_labels.any():
        nodes_to_compute.update(edge_array[0][diff_labels])
        nodes_to_compute.update(edge_array[1][diff_labels])
        for node in nodes_to_compute.copy():
            nodes_to_compute.update(adj[node].indices)
    
    nodes_to_compute = np.array(list(nodes_to_compute))
    
    # 노드별 알파값 계산
    alpha = np.full(num_nodes, fixed_alpha)
    
    # 적응형 유사도 계산
    adaptive_sim = {}
    edges_to_compute = np.arange(edge_array.shape[1])  # 모든 엣지에 대해 계산
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # 미리 계산된 유사도 사용
        sim_structure = structure_similarity[(i, j)]
        sim_location = location_similarity[(i, j)]

        # 적응형 가중치 적용
        # alpha[j]를 사용: j(송신자)의 구조적 신뢰도에 따라 구조적 유사도의 영향력 조절
        # - j의 이웃이 일관된 레이블을 가질 때(alpha[j]가 높을 때) → 구조적 유사도를 더 신뢰
        # - j의 이웃이 다양한 레이블을 가질 때(alpha[j]가 낮을 때) → 위치 유사도에 더 의존
        adaptive_sim[(i, j)] = alpha[j] * sim_structure + (1 - alpha[j]) * sim_location
        
        # a = (alpha[i] + alpha[j]) / 2
        # adaptive_sim[(i, j)] = a * sim_structure + (1 - a) * sim_location
    
    # 통계 계산 및 디버깅을 위한 분포 출력
    computed_alphas = alpha  # 모든 노드의 알파값 사용
    avg_alpha = float(np.mean(computed_alphas))
    dev_alpha = float(np.std(computed_alphas))
    
    return adaptive_sim, avg_alpha, dev_alpha

def save_graph_result(nodes, edges, filename, result_dir='result'):
    """
    Save nodes and edges DataFrame to CSV files in the result directory.
    """
    os.makedirs(result_dir, exist_ok=True)
    nodes_path = os.path.join(result_dir, filename + "_nodes.csv")
    edges_path = os.path.join(result_dir, filename + "_edges.csv")
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

def get_memory_usage():
    """
    현재 프로세스의 메모리 사용량을 다양한 지표로 측정
    """
    import gc
    import psutil
    
    # 가비지 컬렉션 실행
    gc.collect()
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 ** 2,  # Resident Set Size
        'vms': memory_info.vms / 1024 ** 2,  # Virtual Memory Size
        'uss': process.memory_full_info().uss / 1024 ** 2,  # Unique Set Size (Linux only)
        'percent': process.memory_percent()  # 시스템 메모리 대비 비율
    }

# 평균 및 표준편차 계산
def mean_std(arr):
    return np.mean(arr), np.std(arr)

def evaluate_and_save_results(data, pred, result_filename, method_name, times, iters, iter_info=[], result_dir='result'):
    """
    여러 번의 실험 결과(pred)를 받아 각 지표의 평균과 표준편차를 출력 및 파일에 저장
    """
    print(f"# of Nodes: {data.num_nodes}")
    print(f"# of Experiments: {len(pred)}")

    time_mean, time_std = mean_std(times)
    print(f"Elapsed Time: {time_mean:.5f} ± {time_std:.5f} seconds")

    iter_mean, iter_std = mean_std(iters)
    print(f"Iterations: {iter_mean:.5f} ± {iter_std:.5f}")
    
    # 지표별 결과 저장용 리스트
    sn_modularity_scores = []
    # modularity_scores = []
    # conductance_scores = []
    # intra_cluster_distances = []

    cluster_sizes = []
    inter_cluster_distances = []
    density = []

    num_labels_list = [len(np.unique(pred_labels)) for pred_labels in pred]
    nlabel_mean, nlabel_std = mean_std(num_labels_list)
    print(f"# of Labels: {nlabel_mean:.5f} ± {nlabel_std:.5f}")
    
    sigma = 5000.0
    for pred_labels in pred:
        try:
            sn_modularity_score = sn_modularity(data.edge_index, pred_labels, data.rad_x, sigma=sigma)
        except Exception as e:
            print(f"Error calculating SN Modularity: {str(e)}")
            sn_modularity_score = -9
        sn_modularity_scores.append(sn_modularity_score)
    sn_mod_mean, sn_mod_std = mean_std(sn_modularity_scores)
    print(f"SN Modularity: {sn_mod_mean:.5f} ± {sn_mod_std:.5f} (sigma={sigma})")

    for pred_labels in pred:
        # 각 클러스터별 노드 개수를 계산
        unique_labels, counts = np.unique(pred_labels, return_counts=True)
        cluster_sizes.extend(counts)
    cluster_sizes_mean, cluster_sizes_std = mean_std(cluster_sizes)
    print(f"Cluster Sizes: {cluster_sizes_mean:.5f} ± {cluster_sizes_std:.5f}")
    
    for pred_labels in pred:
        inter_cluster_distances.append(inter_cluster_avg_distance(data, pred_labels))
    inter_cluster_distances_mean, inter_cluster_distances_std = mean_std(inter_cluster_distances)
    print(f"Inter-cluster Distances: {inter_cluster_distances_mean:.5f} ± {inter_cluster_distances_std:.5f}")
    
    for pred_labels in pred:
        density.append(density_score(data, pred_labels))
    density_mean, density_std = mean_std(density)
    print(f"Density: {density_mean:.5f} ± {density_std:.5f}")

    # for pred_labels in pred:
    #     try:
    #         modularity_score = modularity(data.edge_index, pred_labels)
    #     except Exception as e:
    #         print(f"Error calculating Modularity: {str(e)}")
    #         modularity_score = -9
    #     modularity_scores.append(modularity_score)
    # mod_mean, mod_std = mean_std(modularity_scores)
    # print(f"Modularity: {mod_mean:.5f} ± {mod_std:.5f}")

    # for pred_labels in pred:
    #     try:
    #         conductance_score = conductance(data.edge_index, pred_labels)
    #     except Exception as e:
    #         print(f"Error calculating Conductance: {str(e)}")
    #         conductance_score = -9
    #     conductance_scores.append(conductance_score)
    # cond_mean, cond_std = mean_std(conductance_scores)
    # print(f"Conductance: {cond_mean:.5f} ± {cond_std:.5f}")

    # for pred_labels in pred:
    #     try:
    #         intra_cluster_distance = intra_cluster_avg_distance(data, pred_labels)
    #     except Exception as e:
    #         print(f"Error calculating Intra-cluster Distance: {str(e)}")
    #         intra_cluster_distance = -9
    #     intra_cluster_distances.append(intra_cluster_distance)
    # intra_mean, intra_std = mean_std(intra_cluster_distances)
    # print(f"Intra-cluster Distance: {intra_mean:.5f} ± {intra_std:.5f}")
    
    # 파일에 저장
    with open(os.path.join(result_dir, result_filename), 'a') as f:
        f.write("\n")
        f.write(f"{method_name}\n")
        # for info in iter_info:
        #     f.write(f"{info}\n")
        # f.write(f"# of Nodes: {data.num_nodes}\n")
        f.write(f"# of Experiments: {len(pred)}\n")
        f.write(f"Elapsed Time: {time_mean:.5f} ± {time_std:.5f} seconds\n")
        f.write(f"Iterations: {iter_mean:.5f} ± {iter_std:.5f}\n")
        f.write(f"# of Labels: {nlabel_mean:.5f} ± {nlabel_std:.5f}\n")
        f.write(f"SN Modularity: {sn_mod_mean:.5f} ± {sn_mod_std:.5f} (sigma={sigma})\n")
        # f.write(f"Modularity: {mod_mean:.5f} ± {mod_std:.5f}\n")
        # f.write(f"Conductance: {cond_mean:.5f} ± {cond_std:.5f}\n")
        # f.write(f"Intra-cluster Distance: {intra_mean:.5f} ± {intra_std:.5f}\n")
        f.write(f"Cluster Sizes: {cluster_sizes_mean:.5f} ± {cluster_sizes_std:.5f}\n")
        f.write(f"Inter-cluster Distances: {inter_cluster_distances_mean:.5f} ± {inter_cluster_distances_std:.5f}\n")
        f.write(f"Density: {density_mean:.5f} ± {density_std:.5f}\n")
        f.write(f"times: {[float(x) for x in times]}\n")
        f.write(f"iters: {[int(x) for x in iters]}\n")
        f.write(f"sn_modularity: {[float(x) for x in sn_modularity_scores]}\n")
        f.write(f"cluster_sizes: {[int(x) for x in cluster_sizes]}\n")
        f.write(f"inter_cluster_distances: {[float(x) for x in inter_cluster_distances]}\n")
        f.write(f"density: {[float(x) for x in density]}\n")

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

def save_result(data, pred_lp, path, file):
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
    edges = set(zip(edge_array[0], edge_array[1]))
    edges_df = pd.DataFrame(list(edges), columns=['source', 'target']).drop_duplicates()
    save_graph_result(nodes_df, edges_df, file, result_dir=path)


def scipy_sparse_to_torch_sparse(sparse_mtx):
    """
    Convert a scipy.sparse CSR/COO matrix to a torch.sparse_coo_tensor.
    """
    if not hasattr(sparse_mtx, 'tocoo'):
        raise ValueError("Input must be a scipy sparse matrix.")
    coo = sparse_mtx.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    i = torch.LongTensor(indices)
    v = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

# 해시 유틸: 라벨 벡터의 64-bit 해시 (결정적, 충돌 확률 매우 낮음)
def labels_hash(labels: torch.Tensor) -> int:
    lab = labels.view(-1).to(torch.int64)
    n = lab.numel()
    dev = lab.device

    if (not hasattr(labels_hash, "_w")
        or labels_hash._w.device != dev
        or labels_hash._w.numel() != n):
        g = torch.Generator(device=dev); g.manual_seed(0)
        labels_hash._w = torch.randint(1, (1<<61)-1, (n,),
                                       dtype=torch.int64, generator=g, device=dev)
    MOD = (1<<61) - 1
    x = (lab * labels_hash._w).remainder(MOD).sum().remainder(MOD)
    return int(x.item())

def compare_clustering_results(pred_arrays, orders):
    """
    여러 클러스터링 결과 간의 유사도를 계산하는 함수
    
    Args:
        pred_arrays: 클러스터링 결과 배열들의 리스트
        orders: 각 결과에 해당하는 순서/방법명 리스트
    
    Returns:
        comparison_results: 비교 결과 딕셔너리
    """
    comparison_results = {}
    
    # 각 결과의 기본 정보 출력
    for i, pred_array in enumerate(pred_arrays):
        unique_labels = np.unique(pred_array)
        print(f"{orders[i]} 순서: {len(unique_labels)}개 클러스터")
        print(f"  - 레이블 범위: {pred_array.min()} ~ {pred_array.max()}")
        print(f"  - 클러스터 크기 분포: 최대={np.max(np.bincount(pred_array.astype(int)))}, "
              f"최소={np.min(np.bincount(pred_array.astype(int)))}, "
              f"평균={np.mean(np.bincount(pred_array.astype(int))):.1f}")
        print(f"  - 고유 레이블 처음 10개: {unique_labels[:10]}")
    
    print()
    
    # 쌍별 비교
    for i in range(len(pred_arrays)):
        for j in range(i+1, len(pred_arrays)):
            # 원본 ARI (레이블 번호 기준)
            ari_raw = adjusted_rand_score(pred_arrays[i], pred_arrays[j])
            
            # 실제 동일한 클러스터에 속하는 노드 쌍의 비율 계산
            same_cluster_i = 0
            same_cluster_j = 0
            same_both = 0
            total_pairs = 0
            
            # 모든 노드 쌍에 대해 샘플링하여 계산 (너무 많으면 메모리 부족)
            n_nodes = len(pred_arrays[i])
            sample_size = min(10000, n_nodes * (n_nodes - 1) // 2)
            
            np.random.seed(42)  # 재현 가능한 결과를 위해
            node_pairs = np.random.choice(n_nodes, size=(sample_size, 2), replace=True)
            
            for pair in node_pairs:
                if pair[0] == pair[1]:
                    continue
                node1, node2 = pair[0], pair[1]
                same_i = pred_arrays[i][node1] == pred_arrays[i][node2]
                same_j = pred_arrays[j][node1] == pred_arrays[j][node2]
                
                if same_i:
                    same_cluster_i += 1
                if same_j:
                    same_cluster_j += 1
                if same_i and same_j:
                    same_both += 1
                total_pairs += 1
            
            # 클러스터 일치도 계산
            if total_pairs > 0:
                cluster_agreement = same_both / total_pairs
                precision = same_both / same_cluster_j if same_cluster_j > 0 else 0
                recall = same_both / same_cluster_i if same_cluster_i > 0 else 0
            else:
                cluster_agreement = precision = recall = 0
            
            # 결과 저장
            comparison_key = f"{orders[i]}_vs_{orders[j]}"
            comparison_results[comparison_key] = {
                'ari_raw': ari_raw,
                'cluster_agreement': cluster_agreement,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{orders[i]} vs {orders[j]}:")
            print(f"  - ARI (원본): {ari_raw:.4f}")
            print(f"  - 클러스터 일치도: {cluster_agreement:.4f}")
            print(f"  - 정밀도: {precision:.4f}, 재현율: {recall:.4f}")
    
    return comparison_results