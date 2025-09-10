import torch
import networkx as nx
from networkx.algorithms import community
import random
import time

def label_propagation(edge_index, labels, order='random', verbose=False):
    """
    NetworkX 라이브러리를 이용한 Label Propagation 알고리즘을 수행합니다.
    
    Args:
        edge_index: (2, E) 크기의 엣지 인덱스 텐서
        labels: 노드 레이블 (사용되지 않음 - NetworkX가 자동으로 커뮤니티 탐지)
        max_iter: 호환성을 위한 파라미터 (NetworkX에서는 내장 수렴 조건 사용)
        verbose: 상세 출력 여부 (기본값: False)
        seed: 랜덤 시드 (기본값: None, 매번 다른 결과)
    
    Returns:
        pred_labels: 예측된 레이블
        iter_info: 반복 과정 정보
    """
    n = labels.size(0)
    
    # NetworkX 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # edge_index를 NetworkX 엣지로 변환
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, 'cpu') else edge_index
    edges = [(int(edge_array[0, i]), int(edge_array[1, i])) for i in range(edge_array.shape[1])]
    G.add_edges_from(edges)
    
    if verbose:
        print(f"그래프 생성 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
    
    # NetworkX label propagation 수행
    try:
        random_seed = int(time.time() * 1000) % 2**32
        communities = community.asyn_lpa_communities(G, seed=random_seed, order=order)
        communities_list = list(communities)
        
        # 커뮤니티를 레이블로 변환
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        for label_idx, comm in enumerate(communities_list):
            for node in comm:
                pred_labels[node] = label_idx
        
        num_communities = len(communities_list)
        iter_info = [
            f"NetworkX Label Propagation 완료",
            f"발견된 커뮤니티 수: {num_communities}",
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
    
    return pred_labels, iter_info