import numpy as np
import psutil
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import os
import pandas as pd

from src.metrics import avg_clustering_coefficient, conductance, intra_cluster_avg_distance, modularity, normalized_cut, sn_modularity, spatial_silhouette

def _to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def _get_edge_pairs(edge_index):
    """Return set of (u, v) pairs from edge_index (numpy array shape [2, N])."""
    edge_array = _to_numpy(edge_index)
    return set(map(tuple, edge_array.T))

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
        neighbors_v = set(adj[v].indices)
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        if union > 0:
            jaccard[(u, v)] = intersection / union

    print("Jaccard similarity computation completed")
    return jaccard

def compute_location_similarity(data, edge_index=None):
    """
    Compute location similarity for connected edges (or given edge_index pairs) using haversine distance.
    Returns: dict with (u, v) tuple as key and location similarity as value.
    """
    print("Computing location similarity...")
    if edge_index is None:
        edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    edge_array = _to_numpy(edge_index)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    
    # 위도/경도 좌표 추출
    coords = _to_numpy(data.rad_x)
    
    # Haversine 거리 계산 함수
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # 지구의 반지름 (km)
        
        # 라디안으로 변환
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine 공식
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # 거리를 유사도로 변환 (거리가 멀수록 유사도는 감소)
        # 지수 감소 함수를 사용하여 거리가 0일 때 1, 멀어질수록 0에 가까워지게 함
        similarity = np.exp(-distance)
        return similarity
    
    # 각 엣지에 대해 유사도 계산
    similarities = {}
    for i, j in zip(edge_array[0], edge_array[1]):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        sim = haversine_distance(lat1, lon1, lat2, lon2)
        similarities[(i, j)] = sim
        similarities[(j, i)] = sim  # 무방향 그래프이므로 양방향 모두 저장
    
    print("Location similarity computation completed")
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

def compute_adaptive_similarity(data, structure_similarity, location_similarity, pred_labels=None, mask=None):
    """
    적응형 유사도 계산 함수
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        structure_similarity: 미리 계산된 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        location_similarity: 미리 계산된 위치 기반 유사도 딕셔너리 {(u,v): similarity_score}
        pred_labels: 현재 예측된 레이블 (기본값: None, data.y 사용)
        mask: 레이블이 있는 노드를 나타내는 마스크 (기본값: None)
    
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
    
    # mask된 노드들 처리
    if mask is not None:
        mask = _to_numpy(mask)
        masked_nodes = np.where(mask)[0]
        nodes_to_compute.update(masked_nodes)
        for node in masked_nodes:
            nodes_to_compute.update(adj[node].indices)
    
    nodes_to_compute = np.array(list(nodes_to_compute))
    
    # 노드별 알파값 계산
    alpha = np.ones(num_nodes)
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
        
        # 실제 엔트로피 계산
        probs = label_counts / len(valid_neigh)
        entropy = -np.sum(probs * np.log(probs))
        
        # 정규화된 엔트로피 (0~1 사이 값)
        normalized_entropy = entropy / max_entropy if max_entropy != 0 else 0
        normalized_entropy = np.clip(normalized_entropy, 0, 1)
        
        # 알파값 계산: 이웃들이 모두 같은 레이블이면 alpha=1, 모두 다른 레이블이면 alpha=0
        alpha[node] = 1 - normalized_entropy

    # Fixed Alpha 실험
    # alpha = np.full(num_nodes, 1.0)
    
    # 적응형 유사도 계산
    adaptive_sim = {}
    edge_mask = np.isin(edge_array[0], nodes_to_compute) | np.isin(edge_array[1], nodes_to_compute)
    edges_to_compute = np.where(edge_mask)[0]
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # 미리 계산된 유사도 사용
        sim_structure = structure_similarity.get((i, j), 0.0)
        sim_location = location_similarity.get((i, j), 0.0)
        
        # 적응형 가중치 적용
        a = (alpha[i] + alpha[j]) / 2
        adaptive_sim[(i, j)] = a * sim_structure + (1 - a) * sim_location
    
    # 통계 계산
    computed_alphas = alpha[nodes_to_compute]
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

def evaluate_and_save_results(data, pred_labels, result_filename, method_name, elapsed_time, iter_info=[], result_dir='result'):
    """
    Evaluate clustering results and save to a file.
    """
    print(f"# of Nodes: {data.num_nodes}")
    print(f"Elapsed Time: {elapsed_time:.3f} seconds")
    # memory_usage = get_memory_usage()
    # print(f"Memory Usage:")
    # print(f"  RSS: {memory_usage['rss']:.2f} MB")
    # print(f"  VMS: {memory_usage['vms']:.2f} MB")
    # if 'uss' in memory_usage:  # Linux only
    #     print(f"  USS: {memory_usage['uss']:.2f} MB")
    # print(f"  Memory Usage %: {memory_usage['percent']:.2f}%")

    print(f"# of Labels: {len(np.unique(pred_labels))}")
    # 각 메트릭 계산 시 오류 발생하면 -9 반환
    sigma = 5000.0
    try:
        sn_modularity_score = sn_modularity(data, pred_labels, sigma=sigma)
    except Exception as e:
        print(f"Error calculating SN Modularity: {str(e)}")
        sn_modularity_score = -9
    print(f"SN Modularity: {sn_modularity_score} (sigma={sigma})")
    # sigma = [1, 3, 6, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # sn_modularity_scores = []
    # for s in sigma:
    #     try:
    #         sn_modularity_score = sn_modularity(data, pred_labels, sigma=s)
    #         sn_modularity_scores.append(sn_modularity_score)
    #         print(f"SN Modularity: {sn_modularity_score} (sigma={s})")
    #     except Exception as e:
    #         print(f"Error calculating SN Modularity: {str(e)}")
    #         sn_modularity_scores.append(-9)
    try:
        modularity_score = modularity(data.edge_index, pred_labels)
    except Exception as e:
        print(f"Error calculating Modularity: {str(e)}")
        modularity_score = -9
    print(f"Modularity: {modularity_score}")
    try:
        conductance_score = conductance(data.edge_index, pred_labels)
    except Exception as e:
        print(f"Error calculating Conductance: {str(e)}")
        conductance_score = -9
    print(f"Conductance: {conductance_score}")
    # try:
    #     avg_clustering_coeff = avg_clustering_coefficient(data.edge_index, pred_labels)
    # except Exception as e:
    #     print(f"Error calculating Avg Clustering Coefficient: {str(e)}")
    #     avg_clustering_coeff = -9
    # print(f"Avg Clustering Coefficient: {avg_clustering_coeff}")
    # try:
    #     normalized_cut_score = normalized_cut(data.edge_index, pred_labels)
    # except Exception as e:
    #     print(f"Error calculating Normalized Cut: {str(e)}")
    #     normalized_cut_score = -9
    # print(f"Normalized Cut: {normalized_cut_score}")

    try:
        intra_cluster_distance = intra_cluster_avg_distance(data, pred_labels)
    except Exception as e:
        print(f"Error calculating Intra-cluster Distance: {str(e)}")
        intra_cluster_distance = -9
    print(f"Intra-cluster Distance: {intra_cluster_distance}")

    # 위도/경도 정보 추출 (data.x에서)
    try:
        if hasattr(data, 'rad_x') and data.rad_x is not None and data.rad_x.shape[1] == 2:
            # data.x의 첫 번째 열이 위도, 두 번째 열이 경도
            coordinates = data.rad_x.cpu().numpy() if hasattr(data.rad_x, 'cpu') else data.rad_x
            silhouette_score_value = spatial_silhouette(coordinates, pred_labels, metric='haversine')
        else:
            print("Warning: No valid coordinate data found in data.x. Skipping spatial silhouette calculation.")
            silhouette_score_value = -9
    except Exception as e:
        print(f"Error calculating spatial silhouette: {str(e)}")
        silhouette_score_value = -9
    
    print(f"Silhouette Score: {silhouette_score_value}")

    with open(os.path.join(result_dir, result_filename), 'a') as f:
        f.write("\n")
        f.write(f"{method_name}\n")
        for info in iter_info:
            f.write(f"{info}\n")
        f.write(f"# of Nodes: {data.num_nodes}\n")
        f.write(f"# of Labels: {len(np.unique(pred_labels))}\n")
        f.write(f"SN Modularity: {sn_modularity_score} (sigma={sigma})\n")
        # for s, score in zip(sigma, sn_modularity_scores):
        #     f.write(f"SN Modularity: {score} (sigma={s})\n")
        f.write(f"Modularity: {modularity_score}\n")
        f.write(f"Conductance: {conductance_score}\n")
        # f.write(f"Avg Clustering Coefficient: {avg_clustering_coeff}\n")
        # f.write(f"Normalized Cut: {normalized_cut_score}\n")
        f.write(f"Intra-cluster Distance: {intra_cluster_distance}\n")
        f.write(f"Silhouette Score: {silhouette_score_value}\n")
        f.write(f"Elapsed Time: {elapsed_time:.3f} seconds\n")
        # f.write(f"Memory Usage:")
        # f.write(f"  RSS: {memory_usage['rss']:.2f} MB\n")
        # f.write(f"  VMS: {memory_usage['vms']:.2f} MB\n")
        # if 'uss' in memory_usage:  # Linux only
        #     f.write(f"  USS: {memory_usage['uss']:.2f} MB\n")
        # f.write(f"  Memory Usage %: {memory_usage['percent']:.2f}%\n")

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