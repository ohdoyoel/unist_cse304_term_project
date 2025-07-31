from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import numpy as np
import networkx as nx
import torch

def evaluate_clustering(true_labels, pred_labels):
    metrics = {}
    metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    metrics['Homogeneity'] = homogeneity_score(true_labels, pred_labels)
    metrics['Completeness'] = completeness_score(true_labels, pred_labels)
    metrics['V-Measure'] = v_measure_score(true_labels, pred_labels)
    return metrics

def _to_numpy(tensor):
    # Helper to convert torch tensor to numpy array
    return tensor.cpu().numpy()

def _build_graph(edge_index):
    # Helper to build nx.Graph from edge_index torch tensor
    import torch
    G = nx.Graph()
    edges = _to_numpy(edge_index).T
    G.add_edges_from(edges)
    return G

def sn_modularity(data, pred_labels, sigma=1.0):
    """
    Spatially-near (SN) modularity를 계산하는 함수
    
    Args:
        data: PyG Data 객체 (edge_index, edge_weight, rad_x 속성 필요)
        pred_labels: 예측된 커뮤니티 레이블
        sigma: 거리 스케일링 파라미터 (기본값: 1.0)
    
    Returns:
        float: SN-modularity 값
    """
    import torch
    
    # 엣지 정보 추출
    edge_index = data.edge_index
    edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else torch.ones(edge_index.size(1))
    pos = data.rad_x  # 노드의 위치 정보 (latitude, longitude in radians)
    
    # 전체 엣지 가중치의 합 (2m)
    total_weight = edge_weight.sum()
    
    # 각 노드의 degree (ki) 계산
    degrees = torch.zeros(pred_labels.size(0))
    for i in range(edge_index.size(1)):
        degrees[edge_index[0, i]] += edge_weight[i]
        degrees[edge_index[1, i]] += edge_weight[i]
    
    # 유니크한 커뮤니티 레이블 추출
    communities = torch.unique(pred_labels)
    
    modularity = 0.0
    for c in communities:
        # 현재 커뮤니티에 속한 노드들의 인덱스
        c_mask = (pred_labels == c)
        c_nodes = torch.where(c_mask)[0]
        
        if len(c_nodes) == 0:
            continue
        
        # 커뮤니티 내부 엣지의 가중치 합 계산 (벡터화)
        edge_mask = c_mask[edge_index[0]] & c_mask[edge_index[1]]
        internal_edges = edge_weight[edge_mask].sum()
        
        # 커뮤니티 내 노드들의 degree 합의 제곱으로 degree_product_sum 계산
        community_degree_sum = degrees[c_nodes].sum()
        degree_product_sum = community_degree_sum * community_degree_sum
        
        # 커뮤니티 중심 계산
        center = pos[c_nodes].mean(dim=0)
        
        # 거리 기반 정규화 항 계산 (Haversine, 벡터화)
        dlat = center[0] - pos[c_nodes, 0]
        dlon = center[1] - pos[c_nodes, 1]
        
        # Haversine 공식 (수치 안정성을 위해 torch.clamp 사용)
        a = torch.sin(dlat/2)**2 + torch.cos(pos[c_nodes, 0]) * torch.cos(center[0]) * torch.sin(dlon/2)**2
        a = torch.clamp(a, 0, 1)  # 수치 안정성을 위해 0과 1 사이로 제한
        distances = 2 * torch.asin(torch.sqrt(a))  # 라디안 단위의 거리
        
        # 거리에 따른 가중치 계산 (거리가 0일 때 1이 되도록)
        distance_term = 1 + ((distances / sigma) ** 2).max() # or max()
        distance_term = torch.clamp(distance_term, min=1e-10)  # 0으로 나누기 방지
        
        # 현재 커뮤니티의 modularity 계산
        contribution = (internal_edges - degree_product_sum / (2 * total_weight)) / distance_term
        modularity += contribution
    
    # 전체 modularity 정규화
    modularity /= (2 * total_weight)
    
    return float(modularity)

def modularity(edge_index, pred_labels):
    import networkx as nx
    
    # edge_index로 그래프 생성
    edges = edge_index.cpu().numpy().T
    G = nx.Graph()
    
    # 실제 존재하는 노드만으로 그래프 생성
    unique_nodes = set(edges.flatten())
    node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
    
    # 엣지 리매핑
    remapped_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    G.add_edges_from(remapped_edges)
    
    # 예측된 레이블로 커뮤니티 생성 (리매핑된 노드 번호 사용)
    communities = []
    for label in torch.unique(pred_labels):
        nodes = (pred_labels == label).nonzero().flatten().cpu().numpy()
        # 노드 번호 리매핑
        remapped_nodes = [node_to_idx[n] for n in nodes if n in node_to_idx]
        if remapped_nodes:  # 빈 커뮤니티 제외
            communities.append(remapped_nodes)
    
    return nx.community.modularity(G, communities)

def conductance(edge_index, labels):
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    conductances = []
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        if len(nodes) == 0 or len(nodes) == G.number_of_nodes():
            continue
        cut_size = nx.cut_size(G, nodes)
        volume = nx.volume(G, nodes)
        if volume > 0:
            conductances.append(cut_size / volume)
    return float(np.mean(conductances)) if conductances else 0.0

def avg_clustering_coefficient(edge_index, labels, sample_size=10000):
    """
    샘플링을 통해 클러스터링 계수를 계산합니다.
    
    Parameters:
    -----------
    edge_index : torch.Tensor or numpy.ndarray
        엣지 리스트 [2, E]
    labels : torch.Tensor or numpy.ndarray
        노드 레이블
    sample_size : int
        각 클러스터당 샘플링할 최대 노드 수
    """
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    avg_coeffs = []
    
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        if len(nodes) < 3:  # 클러스터링 계수를 계산하려면 최소 3개의 노드가 필요
            continue
            
        # 노드 샘플링
        if len(nodes) > sample_size:
            nodes = np.random.choice(nodes, sample_size, replace=False)
            
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_edges() == 0:  # 엣지가 없는 경우
            continue
            
        try:
            coeff = nx.average_clustering(subgraph)
            if not np.isnan(coeff):  # NaN이 아닌 경우만 추가
                avg_coeffs.append(coeff)
        except:
            continue
            
    return float(np.mean(avg_coeffs)) if avg_coeffs else 0.0

def normalized_cut(edge_index, labels, sample_size=1000):
    """
    샘플링을 통해 normalized cut을 계산합니다.
    
    Parameters:
    -----------
    edge_index : torch.Tensor or numpy.ndarray
        엣지 리스트 [2, E]
    labels : torch.Tensor or numpy.ndarray
        노드 레이블
    sample_size : int
        각 클러스터당 샘플링할 최대 노드 수
    """
    import networkx as nx
    
    G = _build_graph(edge_index)
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    ncuts = []
    
    for label in unique_labels:
        # 현재 클러스터의 노드들
        nodes = np.where(labels_np == label)[0]
        if len(nodes) == 0:
            continue
            
        # 노드 샘플링
        if len(nodes) > sample_size:
            nodes = np.random.choice(nodes, sample_size, replace=False)
            
        # 나머지 노드들도 샘플링
        other_nodes = np.where(labels_np != label)[0]
        if len(other_nodes) > sample_size:
            other_nodes = np.random.choice(other_nodes, sample_size, replace=False)
            
        # 전체 샘플링된 노드
        sampled_nodes = np.concatenate([nodes, other_nodes])
        subgraph = G.subgraph(sampled_nodes)
        
        # 샘플링된 서브그래프에서의 레이블
        subgraph_labels = labels_np[sampled_nodes]
        current_nodes = sampled_nodes[subgraph_labels == label]
        
        try:
            # normalized cut 계산
            cut_value = nx.cut_size(subgraph, current_nodes)
            vol_A = nx.volume(subgraph, current_nodes)
            vol_B = nx.volume(subgraph, set(subgraph.nodes) - set(current_nodes))
            
            if vol_A > 0 and vol_B > 0:  # 0으로 나누기 방지
                ncut = (cut_value / vol_A) + (cut_value / vol_B)
                ncuts.append(ncut)
        except:
            continue
            
    return float(np.mean(ncuts)) if ncuts else 0.0

def intra_cluster_avg_distance(data, pred_labels, max_sample_size=1000):
    """
    각 클러스터 내 노드들 간의 평균 거리 계산
    
    Args:
        data: 데이터 객체 (rad_x: 라디안 단위의 [위도, 경도] 좌표 포함)
        pred_labels: 예측된 클러스터 레이블 배열
        max_sample_size: 큰 클러스터에서 사용할 최대 샘플 크기
        
    Returns:
        float: 클러스터 내 평균 거리 (km)
    """
    import numpy as np
    from collections import defaultdict
    
    # 데이터 준비 (이미 라디안 단위)
    rad_coords = data.rad_x.cpu().numpy() if hasattr(data.rad_x, 'cpu') else data.rad_x
    pred_labels = pred_labels.cpu().numpy() if hasattr(pred_labels, 'cpu') else pred_labels
    
    # 클러스터별 노드 그룹화
    clusters = defaultdict(list)
    for node_idx, label in enumerate(pred_labels):
        clusters[label].append(node_idx)
    
    total_distance = 0.0
    total_pairs = 0
    
    def calculate_distances_batch(coords, batch_size=10000):
        """배치 단위로 거리 계산"""
        n = len(coords)
        distances = []
        pairs_count = 0
        
        for i in range(0, n, batch_size):
            batch_coords1 = coords[i:min(i+batch_size, n)]
            
            # 현재 배치와 나머지 모든 좌표 사이의 거리 계산
            lats1 = batch_coords1[:, 0].reshape(-1, 1)
            lons1 = batch_coords1[:, 1].reshape(-1, 1)
            
            for j in range(i, n, batch_size):
                batch_coords2 = coords[j:min(j+batch_size, n)]
                lats2 = batch_coords2[:, 0].reshape(1, -1)
                lons2 = batch_coords2[:, 1].reshape(1, -1)
                
                dlat = lats2 - lats1
                dlon = lons2 - lons1
                
                a = np.sin(dlat/2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                batch_distances = 6371 * c
                
                # 상삼각 행렬만 사용 (중복 제거)
                if i == j:
                    mask = np.triu(np.ones_like(batch_distances, dtype=bool), k=1)
                    batch_distances = batch_distances[mask]
                else:
                    batch_distances = batch_distances.flatten()
                
                distances.extend(batch_distances)
                pairs_count += len(batch_distances)
                
        return np.array(distances), pairs_count
    
    # 각 클러스터에 대해
    for label, nodes in clusters.items():
        n_nodes = len(nodes)
        
        if n_nodes < 2:  # 노드가 1개인 클러스터는 건너뛰기
            continue
            
        # 큰 클러스터의 경우 샘플링
        if n_nodes > max_sample_size:
            sampled_indices = np.random.choice(nodes, max_sample_size, replace=False)
            coords = rad_coords[sampled_indices]
        else:
            coords = rad_coords[nodes]
        
        # 배치 단위로 거리 계산
        cluster_distances, pairs_count = calculate_distances_batch(coords)
        
        if len(cluster_distances) > 0:
            total_distance += np.sum(cluster_distances)
            total_pairs += pairs_count
    
    # 전체 평균 계산
    if total_pairs == 0:
        return 0.0
        
    avg_distance = total_distance / total_pairs
    
    return avg_distance

def spatial_silhouette(coordinates, labels, metric='euclidean', sample_size=10000):
    """
    좌표 정보를 사용하여 실루엣 스코어를 계산합니다.
    
    Args:
        coordinates: shape (2, N) 또는 (N, 2)의 노드 좌표 배열
        labels: shape (N,)의 클러스터 레이블 배열
        metric: 거리 계산 방식 ('euclidean', 'manhattan', 'haversine')
        sample_size: 샘플링할 노드 수. 전체 노드 수보다 크면 전체 데이터 사용
    
    Returns:
        float: 공간적 실루엣 스코어
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # coordinates shape 확인 및 변환
    if coordinates.shape[0] == 2:
        coordinates = coordinates.T  # (2, N) -> (N, 2)
    
    n_samples = len(coordinates)
    if sample_size < n_samples:
        # 각 클러스터에서 비례하여 샘플링
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        sample_indices = []
        
        for label, count in zip(unique_labels, label_counts):
            label_indices = np.where(labels == label)[0]
            # 각 클러스터의 크기에 비례하여 샘플 수 결정
            n_samples_label = int(sample_size * (count / n_samples))
            # 최소 1개는 샘플링
            n_samples_label = max(1, n_samples_label)
            # 랜덤 샘플링
            sampled = np.random.choice(
                label_indices, 
                size=min(n_samples_label, len(label_indices)), 
                replace=False
            )
            sample_indices.extend(sampled)
            
        coordinates = coordinates[sample_indices]
        labels = labels[sample_indices]
    
    if metric == 'haversine':
        from sklearn.metrics.pairwise import haversine_distances
        # 거리 행렬 계산 (단위: 라디안)
        distances = haversine_distances(coordinates)
        return silhouette_score(distances, labels, metric='precomputed')
    else:
        return silhouette_score(coordinates, labels, metric=metric)

def graph_density(edge_index, num_nodes):
    G = _build_graph(edge_index)
    return nx.density(G)