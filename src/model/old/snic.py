import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from typing import Tuple, List, Dict, Set
from collections import defaultdict

def calculate_community_center(nodes: Set[int], rad_coords: np.ndarray) -> np.ndarray:
    """커뮤니티의 중심점 계산"""
    return np.mean(rad_coords[list(nodes)], axis=0)

def calculate_community_contribution(G: nx.Graph, 
                                  nodes: Set[int],
                                  center: np.ndarray,
                                  rad_coords: np.ndarray,
                                  k_i: Dict[int, int],
                                  m: int,
                                  sigma: float) -> float:
    """단일 커뮤니티의 modularity 기여도 계산"""
    if len(nodes) < 2:
        return 0.0
        
    # 네트워크 모듈성 항
    edges_inside = sum(1 for i in nodes for j in (set(G.neighbors(i)) & nodes))
    degree_sum = sum(k_i[i] for i in nodes)
    degree_product_sum = degree_sum * degree_sum - sum(k_i[i] * k_i[i] for i in nodes)
    community_term = edges_inside - degree_product_sum / (4 * m)
    
    # 거리 계산 (벡터화)
    node_coords = rad_coords[list(nodes)]
    dlat = node_coords[:, 0] - center[0]
    dlon = node_coords[:, 1] - center[1]
    a = np.sin(dlat/2)**2 + np.cos(node_coords[:, 0]) * np.cos(center[0]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = 6371 * c / sigma
    
    max_squared_dist = np.max(distances ** 2) if len(distances) > 0 else 0
    return community_term / (1 + max_squared_dist)

def compute_modularity_change(G: nx.Graph,
                            node: int,
                            old_comm: int,
                            new_comm: int,
                            comm_to_nodes: Dict[int, Set[int]],
                            rad_coords: np.ndarray,
                            k_i: Dict[int, int],
                            m: int,
                            sigma: float) -> float:
    """노드 이동에 따른 modularity 변화량 계산"""
    old_nodes = comm_to_nodes[old_comm]
    new_nodes = comm_to_nodes[new_comm]
    
    # 이전 상태의 기여도
    old_comm_center = calculate_community_center(old_nodes, rad_coords)
    new_comm_center = calculate_community_center(new_nodes, rad_coords)
    old_contribution = calculate_community_contribution(G, old_nodes, old_comm_center, rad_coords, k_i, m, sigma)
    new_contribution = calculate_community_contribution(G, new_nodes, new_comm_center, rad_coords, k_i, m, sigma)
    
    # 노드 이동 후 상태의 기여도
    old_nodes_after = old_nodes - {node}
    new_nodes_after = new_nodes | {node}
    old_comm_center_after = calculate_community_center(old_nodes_after, rad_coords) if old_nodes_after else np.zeros(2)
    new_comm_center_after = calculate_community_center(new_nodes_after, rad_coords)
    old_contribution_after = calculate_community_contribution(G, old_nodes_after, old_comm_center_after, rad_coords, k_i, m, sigma)
    new_contribution_after = calculate_community_contribution(G, new_nodes_after, new_comm_center_after, rad_coords, k_i, m, sigma)
    
    return (old_contribution_after + new_contribution_after - old_contribution - new_contribution) / (2 * m)

def snic_method(data, sigma: float = 5000.0, max_iter: int = 100) -> np.ndarray:
    """SNIC (Spatially-Near Community Detection) 알고리즘 구현"""
    if not hasattr(data, 'rad_x'):
        raise ValueError("데이터에 rad_x (라디안 단위의 위도/경도) 좌표가 필요합니다")
    
    # 데이터 준비
    rad_coords = data.rad_x.cpu().numpy() if hasattr(data.rad_x, 'cpu') else data.rad_x
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    
    # 그래프 생성 및 기본 정보 계산
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(zip(edge_array[0], edge_array[1]))
    k_i = dict(G.degree())
    m = G.number_of_edges()
    
    # 초기 파티션 설정
    node_to_comm = {node: node for node in range(data.num_nodes)}
    comm_to_nodes = {i: {i} for i in range(data.num_nodes)}
    
    for i in range(max_iter):
        improved = False
        total_moves = 0
        
        # 각 노드에 대해
        for node in range(data.num_nodes):
            current_comm = node_to_comm[node]
            best_gain = 0
            best_comm = current_comm
            
            # 이웃 커뮤니티 수집
            neighbor_comms = {node_to_comm[neigh] for neigh in G.neighbors(node)}
            neighbor_comms.discard(current_comm)
            
            # 각 이웃 커뮤니티로 이동했을 때의 이득 계산
            for target_comm in neighbor_comms:
                gain = compute_modularity_change(G, node, current_comm, target_comm,
                                              comm_to_nodes, rad_coords, k_i, m, sigma)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = target_comm
            
            # 실제로 가장 좋은 커뮤니티로 이동
            if best_gain > 0:
                improved = True
                total_moves += 1
                
                # 노드 이동
                comm_to_nodes[current_comm].remove(node)
                comm_to_nodes[best_comm].add(node)
                node_to_comm[node] = best_comm
                
                # 빈 커뮤니티 제거
                if not comm_to_nodes[current_comm]:
                    del comm_to_nodes[current_comm]
        
        print(f"Iter {i+1}: Total moves = {total_moves}")
        if not improved:
            break
    
    # 결과 변환
    result = np.zeros(data.num_nodes, dtype=int)
    for new_label, (_, nodes) in enumerate(comm_to_nodes.items()):
        for node in nodes:
            result[node] = new_label
            
    return result