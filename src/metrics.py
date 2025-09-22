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
    G = nx.Graph()
    edges = _to_numpy(edge_index).T
    G.add_edges_from(edges)
    return G

def sn_modularity(edge_index, pred_labels, rad_x, weight="weight", sigma=1.0):
    """
    Spatially-near (SN) modularity를 계산하는 함수
    
    NetworkX의 modularity 함수와 동일한 인터페이스를 사용하여
    공간적 거리를 고려한 modularity를 계산합니다.
    
    Args:
        edge_index: torch.Tensor or numpy.ndarray
        엣지 리스트 [2, E]
        pred_labels: torch.Tensor or numpy.ndarray
        노드 레이블
        rad_x: torch.Tensor or numpy.ndarray
        노드 위치 (라디안 단위)
        weight: 엣지 가중치 속성명 (기본값: "weight")
        sigma: 거리 스케일링 파라미터 (기본값: 1.0)
    
    Returns:
        float: SN-modularity 값
    """
    import networkx as nx
    import numpy as np
    
    # edge_index로 그래프 생성  
    edges = _to_numpy(edge_index).T
    G = nx.Graph()
    
    # 실제 존재하는 노드만으로 그래프 생성
    unique_nodes = set(edges.flatten())
    node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
    
    # 엣지 리매핑
    remapped_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    G.add_edges_from(remapped_edges)
    
    # 예측된 레이블로 커뮤니티 생성 (리매핑된 노드 번호 사용)
    communities = []
    labels_np = _to_numpy(pred_labels)
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        # 노드 번호 리매핑
        remapped_nodes = [node_to_idx[n] for n in nodes if n in node_to_idx]
        if remapped_nodes:  # 빈 커뮤니티 제외
            communities.append(remapped_nodes)
    
    if not isinstance(communities, list):
        communities = list(communities)
    
    # 위치 정보 추출 (rad_x를 numpy 배열로 변환하고 원래 노드 번호에서 리매핑된 노드 번호로 매핑)
    rad_x_np = _to_numpy(rad_x)
    positions = {}
    for orig_node, remapped_node in node_to_idx.items():
        if orig_node < len(rad_x_np):
            positions[remapped_node] = tuple(rad_x_np[orig_node])
    
    # 전체 가중치 합 계산
    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m**2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2
    
    def haversine_distance(pos1, pos2):
        """두 위치 간의 Haversine 거리 계산 (라디안 단위 입력)"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        a = np.clip(a, 0, 1)  # 수치 안정성
        distance = 2 * np.arcsin(np.sqrt(a))
        
        return distance
    
    def community_contribution(community):
        """각 커뮤니티의 SN-modularity 기여도 계산"""
        comm = set(community)
        
        if len(comm) == 0:
            return 0.0
        
        # 분자 계산: Σ_{i,j∈c} [w_ij - k_i k_j/(2m)]
        numerator = 0.0
        
        # 커뮤니티 내의 모든 엣지에 대해 계산
        for u, v, wt in G.edges(comm, data=weight, default=1):
            if v in comm:  # 커뮤니티 내부 엣지만
                # 실제 엣지 가중치 - 기대 가중치
                expected_weight = (out_degree[u] * out_degree[v]) / (2 * m)
                numerator += wt - expected_weight
        
        # 커뮤니티 중심 계산을 위한 위치 정보 확인
        comm_positions = [positions[node] for node in comm if node in positions]
        if len(comm_positions) == 0:
            # 위치 정보가 없으면 기본 modularity 반환 (distance_term = 1)
            return numerator
        
        # 커뮤니티 중심 계산 (x_c: 평균 위치)
        center_lat = np.mean([pos[0] for pos in comm_positions])
        center_lon = np.mean([pos[1] for pos in comm_positions])
        center = (center_lat, center_lon)
        
        # 분모 계산: 1 + agg_{i∈c}(d(i,x_c)/σ)²
        # 여기서 agg는 sum으로 구현 (논문에 따라 다를 수 있음)
        distance_agg = 0.0
        for node in comm:
            if node in positions:
                distance = haversine_distance(center, positions[node])
                distance_agg += (distance / sigma) ** 2
        
        denominator = 1 + distance_agg
        denominator = max(denominator, 1e-10)  # 0으로 나누기 방지
        
        # 커뮤니티별 기여도: [분자] / [분모]
        return numerator / denominator
    
    # 전체 SN-modularity: (1/2m) × Σ_{c∈C} [각 커뮤니티 기여도]
    total_contribution = sum(map(community_contribution, communities))
    return total_contribution / (2 * m)

def modularity(edge_index, pred_labels, edge_weight=None, weight="weight"):    
    """
    모듈러리티를 계산하는 함수 (가중치 지원)
    
    Args:
        edge_index: torch.Tensor, 엣지 인덱스 [2, E]
        pred_labels: torch.Tensor, 예측된 노드 레이블
        edge_weight: torch.Tensor or None, 엣지 가중치 [E] (기본값: None, 모든 엣지 가중치 1)
        weight: str, 가중치 속성명 (기본값: "weight")
    
    Returns:
        float: 모듈러리티 값
    """
    # edge_index로 그래프 생성
    edges = edge_index.cpu().numpy().T
    G = nx.Graph()
    
    # 실제 존재하는 노드만으로 그래프 생성
    unique_nodes = set(edges.flatten())
    node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
    
    # 엣지 리매핑 및 가중치 정보 추가
    remapped_edges = []
    if edge_weight is not None:
        edge_weights = edge_weight.cpu().numpy()
        for i, (u, v) in enumerate(edges):
            remapped_u, remapped_v = node_to_idx[u], node_to_idx[v]
            # 가중치가 있는 엣지 추가
            remapped_edges.append((remapped_u, remapped_v, {weight: edge_weights[i]}))
    else:
        for u, v in edges:
            remapped_u, remapped_v = node_to_idx[u], node_to_idx[v]
            # 기본 가중치 1로 엣지 추가
            remapped_edges.append((remapped_u, remapped_v, {weight: 1.0}))
    
    G.add_edges_from(remapped_edges)
    
    # 예측된 레이블로 커뮤니티 생성 (리매핑된 노드 번호 사용)
    communities = []
    for label in torch.unique(pred_labels):
        nodes = (pred_labels == label).nonzero().flatten().cpu().numpy()
        # 노드 번호 리매핑
        remapped_nodes = [node_to_idx[n] for n in nodes if n in node_to_idx]
        if remapped_nodes:  # 빈 커뮤니티 제외
            communities.append(remapped_nodes)
    
    # 가중치를 고려한 모듈러리티 계산
    return nx.community.modularity(G, communities, weight=weight)

def conductance(edge_index, labels, edge_weight=None, weight=None):
    """
    컨덕턴스를 계산하는 함수 (가중치 지원)
    
    Args:
        edge_index: torch.Tensor, 엣지 인덱스 [2, E]
        labels: torch.Tensor, 노드 레이블
        edge_weight: torch.Tensor or None, 엣지 가중치 [E] (기본값: None)
        weight: str, 가중치 속성명 (기본값: None, 가중치 없음)
    
    Returns:
        float: 각 클러스터의 컨덕턴스 평균값
    """
    # 가중치가 있는 그래프 생성
    if edge_weight is not None:
        edges = _to_numpy(edge_index).T
        edge_weights = _to_numpy(edge_weight)
        G = nx.Graph()
        
        # 가중치와 함께 엣지 추가
        if weight is None:
            weight = 'weight'  # 기본 가중치 속성명
        
        for i, (u, v) in enumerate(edges):
            G.add_edge(u, v, **{weight: edge_weights[i]})
    else:
        G = _build_graph(edge_index)
    
    labels_np = _to_numpy(labels)
    unique_labels = np.unique(labels_np)
    conductances = []
    
    for label in unique_labels:
        nodes = np.where(labels_np == label)[0]
        # 빈 클러스터나 전체 그래프와 같은 크기의 클러스터는 건너뛰기
        if len(nodes) == 0 or len(nodes) == G.number_of_nodes():
            continue
        
        # 나머지 노드들 (T 집합)
        remaining_nodes = set(G.nodes()) - set(nodes)
        
        if len(remaining_nodes) == 0:
            continue
            
        # NetworkX의 conductance 함수 사용
        cond = nx.conductance(G, nodes, remaining_nodes, weight=weight)
        conductances.append(cond)
    
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

def intra_cluster_avg_distance(data, pred_labels, max_sample_size=10000):
    """
    각 클러스터 내 노드들 간의 평균 거리 계산
    
    Args:
        data: 데이터 객체 (rad_x: 라디안 단위의 [위도, 경도] 좌표 포함)
        pred_labels: 예측된 클러스터 레이블 배열
        
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
    
    def calculate_distances_batch(coords, batch_size=2000):
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

def inter_cluster_avg_distance(data, pred_labels, max_sample_size=1000, max_clusters=500):
    """
    각 클러스터 간의 평균 거리 계산 (최적화된 버전)
    
    Args:
        data: 데이터 객체 (rad_x: 라디안 단위의 [위도, 경도] 좌표 포함)
        pred_labels: 예측된 클러스터 레이블 배열
        max_sample_size: 큰 클러스터에서 샘플링할 최대 노드 수 (기본값 감소)
        max_clusters: 계산할 최대 클러스터 수 (기본값 크게 감소)
        
    Returns:
        float: 클러스터 간 평균 거리 (km)
    """
    import numpy as np
    from collections import Counter
    
    # 데이터 준비 (이미 라디안 단위)
    rad_coords = data.rad_x.cpu().numpy() if hasattr(data.rad_x, 'cpu') else data.rad_x
    pred_labels = pred_labels.cpu().numpy() if hasattr(pred_labels, 'cpu') else pred_labels
    
    # Counter를 사용해 클러스터 크기를 빠르게 계산
    cluster_counts = Counter(pred_labels)
    
    # 클러스터가 2개 미만이면 계산 불가
    if len(cluster_counts) < 2:
        return 0.0
    
    # 크기가 1인 클러스터들을 제거 (노이즈 제거)
    valid_clusters = {label: count for label, count in cluster_counts.items() if count > 1}
    
    if len(valid_clusters) < 2:
        return 0.0
    
    # 상위 max_clusters개의 큰 클러스터만 선택
    if len(valid_clusters) > max_clusters:
        # 가장 큰 클러스터들만 선택
        sorted_clusters = sorted(valid_clusters.items(), key=lambda x: x[1], reverse=True)
        selected_clusters = dict(sorted_clusters[:max_clusters])
    else:
        selected_clusters = valid_clusters
    
    cluster_ids = list(selected_clusters.keys())
    
    # 벡터화된 방식으로 클러스터별 인덱스 생성
    cluster_mask = np.isin(pred_labels, cluster_ids)
    filtered_labels = pred_labels[cluster_mask]
    filtered_coords = rad_coords[cluster_mask]
    
    # 각 클러스터의 중심점을 벡터화된 방식으로 계산
    cluster_centers = {}
    
    for label in cluster_ids:
        # 해당 클러스터의 좌표들 선택
        mask = filtered_labels == label
        cluster_coords = filtered_coords[mask]
        
        # 샘플링 (필요한 경우)
        if len(cluster_coords) > max_sample_size:
            # 랜덤 샘플링 대신 균등 샘플링 사용 (더 빠름)
            indices = np.linspace(0, len(cluster_coords)-1, max_sample_size, dtype=int)
            cluster_coords = cluster_coords[indices]
        
        # 중심점 계산 (벡터화된 평균)
        cluster_centers[label] = np.mean(cluster_coords, axis=0)
    
    # 모든 중심점을 배열로 변환
    centers_array = np.array(list(cluster_centers.values()))
    
    # 벡터화된 Haversine 거리 계산
    def vectorized_haversine(centers):
        """벡터화된 Haversine 거리 계산"""
        n = len(centers)
        if n < 2:
            return np.array([])
        
        # 모든 쌍의 조합을 생성
        idx1, idx2 = np.triu_indices(n, k=1)
        
        lat1, lon1 = centers[idx1, 0], centers[idx1, 1]
        lat2, lon2 = centers[idx2, 0], centers[idx2, 1]
        
        # Haversine 공식 (벡터화됨)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = 6371 * c  # 지구 반지름 6371km
        
        return distances
    
    # 모든 클러스터 간 거리를 한 번에 계산
    distances = vectorized_haversine(centers_array)
    
    if len(distances) == 0:
        return 0.0
    
    # 평균 거리 반환
    return np.mean(distances)

def density_score(data, pred_labels):
    """
    클러스터별 밀도 스코어를 계산하는 함수 (최적화된 버전)
    
    Args:
        data: Data 객체 (edge_index 필요)
        pred_labels: 예측된 노드 레이블
    
    Returns:
        float: 모든 클러스터의 평균 밀도 스코어
    """
    # 데이터 타입을 numpy로 변환
    edge_index = _to_numpy(data.edge_index)
    labels = _to_numpy(pred_labels)
    
    # 각 클러스터별 엣지 개수를 한 번에 계산
    u_labels = labels[edge_index[0]]  # 출발 노드들의 라벨
    v_labels = labels[edge_index[1]]  # 도착 노드들의 라벨
    
    # 같은 클러스터 내의 엣지들만 필터링
    same_cluster_mask = (u_labels == v_labels)
    intra_cluster_edges = edge_index[:, same_cluster_mask]
    
    # 각 클러스터별 엣지 개수 카운트
    if intra_cluster_edges.shape[1] > 0:
        edge_labels = labels[intra_cluster_edges[0]]  # 클러스터 내 엣지들의 라벨
        unique_edge_labels, edge_counts = np.unique(edge_labels, return_counts=True)
        cluster_edge_dict = dict(zip(unique_edge_labels, edge_counts))
    else:
        cluster_edge_dict = {}
    
    # 각 클러스터별 노드 개수 계산
    unique_labels, node_counts = np.unique(labels, return_counts=True)
    
    # 각 클러스터의 밀도 계산
    cluster_densities = []
    for i, label in enumerate(unique_labels):
        cluster_size = node_counts[i]
        
        if cluster_size <= 1:
            # 노드가 1개 이하면 밀도는 0
            cluster_densities.append(0.0)
            continue
        
        # 클러스터 내 엣지 개수 가져오기
        cluster_edges = cluster_edge_dict.get(label, 0)
        
        # 밀도 계산: (클러스터 내 엣지 개수) / (클러스터 내 노드 개수)
        density = cluster_edges / cluster_size
        cluster_densities.append(density)
    
    # 모든 클러스터의 평균 밀도 반환
    return np.mean(cluster_densities) if cluster_densities else 0.0

def spatial_silhouette(coordinates, labels, metric='euclidean', sample_size=5000):
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