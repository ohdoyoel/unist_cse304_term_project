import torch
import networkx as nx
from src.model.label_propagation import custom_asyn_lpa_communities
from src.utils import compute_fixed_alpha_similarity
import time

def fixed_alpha_label_propagation(data, labels, fixed_alpha, structure_similarity=None, location_similarity=None, verbose=False):
    """
    NetworkX 라이브러리를 이용한 Fixed Alpha Label Propagation 알고리즘을 수행합니다.
    구조 유사도와 위치 유사도를 fixed_alpha 비율로 배합하여 가중치로 사용합니다.
    
    Args:
        data: 그래프 데이터 객체
        labels: 노드 레이블 (사용되지 않음 - NetworkX가 자동으로 커뮤니티 탐지)
        fixed_alpha: 구조 유사도와 위치 유사도의 배합 비율 (0~1)
        structure_similarity: 구조 유사도 딕셔너리
        location_similarity: 위치 유사도 딕셔너리
        max_iter: 호환성을 위한 파라미터 (NetworkX에서는 내장 수렴 조건 사용)
        verbose: 상세 출력 여부 (기본값: False)
    
    Returns:
        pred_labels: 예측된 레이블
        mixed_similarity: 배합된 유사도 딕셔너리
        iter_info: 반복 과정 정보
    """
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity와 location_similarity는 반드시 제공되어야 합니다.")

    n = labels.size(0)
    
    # fixed_alpha 비율로 구조 유사도와 위치 유사도를 배합
    mixed_similarity = {}
    all_edges = set(structure_similarity.keys()) | set(location_similarity.keys())
    
    for edge in all_edges:
        struct_sim = structure_similarity.get(edge, 0.0)
        loc_sim = location_similarity.get(edge, 0.0)
        # fixed_alpha: 구조 유사도 가중치, (1-fixed_alpha): 위치 유사도 가중치
        mixed_sim = fixed_alpha * struct_sim + (1 - fixed_alpha) * loc_sim
        if mixed_sim > 0:  # 양수인 유사도만 사용
            mixed_similarity[edge] = mixed_sim
    
    if verbose:
        print(f"Fixed Alpha: {fixed_alpha}")
        print(f"총 {len(mixed_similarity)}개의 가중치 있는 엣지 생성")
        print(f"평균 가중치: {sum(mixed_similarity.values()) / len(mixed_similarity):.6f}")
    
    # NetworkX 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # 가중치가 있는 엣지 추가
    weighted_edges = []
    for (u, v), weight in mixed_similarity.items():
        weighted_edges.append((u, v, weight))
    
    G.add_weighted_edges_from(weighted_edges)
    
    if verbose:
        print(f"그래프 생성 완료: 노드 {G.number_of_nodes()}개, 가중치 엣지 {G.number_of_edges()}개")
    
    # NetworkX asynchronous label propagation 수행
    try:
        random_seed = int(time.time() * 1000) % 2**32
        communities, it = custom_asyn_lpa_communities(G, weight='weight', seed=random_seed)
        communities_list = list(communities)
        
        # 커뮤니티를 레이블로 변환
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        for label_idx, comm in enumerate(communities_list):
            for node in comm:
                pred_labels[node] = label_idx
        
        num_communities = len(communities_list)
        iter_info = [
            f"NetworkX Fixed Alpha Label Propagation 완료",
            f"발견된 커뮤니티 수: {num_communities}",
            f"Fixed Alpha: {fixed_alpha}",
            f"사용된 가중치 엣지 수: {len(mixed_similarity)}"
        ]
        
        if verbose:
            for info in iter_info:
                print(info)
            print(f"각 커뮤니티 크기: {[len(comm) for comm in communities_list]}")
        
    except Exception as e:
        if verbose:
            print(f"NetworkX label propagation 실행 중 오류: {e}")
        # 폴백: 모든 노드를 하나의 커뮤니티로 할당
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        iter_info = [f"오류로 인해 단일 커뮤니티로 할당: {e}"]
    
    return pred_labels, mixed_similarity, it
