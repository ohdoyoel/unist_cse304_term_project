import torch

def label_propagation(edge_index, labels, max_iter=1000, verbose=True):
    """
    Majority voting 기반의 Label Propagation 알고리즘을 수행합니다.
    
    Args:
        edge_index: (2, E) 크기의 엣지 인덱스 텐서
        labels: 노드 레이블
        max_iter: 최대 반복 횟수 (기본값: 1000)
        verbose: 상세 출력 여부 (기본값: True)
    
    Returns:
        pred_labels: 예측된 레이블
        iter_info: 반복 과정 정보
    """
    n = labels.size(0)
    prev_unique_labels = n  # 초기값 설정
    prev_changes = n  # 초기값 설정
    
    # edge_index로부터 인접 리스트 생성
    adj_list = [[] for _ in range(n)]
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, 'cpu') else edge_index
    for i in range(edge_array.shape[1]):
        u, v = edge_array[0, i], edge_array[1, i]
        adj_list[u].append(v)
        adj_list[v].append(u)  # 무방향 그래프 가정
    
    # 초기 레이블 설정
    pred_labels = labels.clone()
    
    iter_info = []
    
    for iter_idx in range(max_iter):
        pred_labels_new = pred_labels.clone()
        label_changes = 0
        
        # 각 노드에 대해 majority voting 수행
        for node in range(n):         
            if not adj_list[node]:  # 이웃이 없는 노드는 건너뜀
                continue
                
            # 이웃들의 레이블 수집
            neighbor_labels = pred_labels[adj_list[node]]
            # 가장 많이 등장하는 레이블 선택
            values, counts = torch.unique(neighbor_labels, return_counts=True)
            pred_labels_new[node] = values[counts.argmax()]
            
            if pred_labels_new[node] != pred_labels[node]:
                label_changes += 1
        
        num_unique_labels = len(torch.unique(pred_labels_new))
        iter_info.append(f"Iter {iter_idx+1}: #labels={num_unique_labels}, changes={label_changes}")
        if verbose:
            print(iter_info[-1])
        
        # 수렴 조건 체크
        if (
            label_changes / n < 0.01
            or (num_unique_labels >= prev_unique_labels and label_changes >= prev_changes)
        ):
            iter_info.append(f"Converged at iteration {iter_idx+1}")
            if verbose:
                print(iter_info[-1])
            break
            
        prev_unique_labels = num_unique_labels
        prev_changes = label_changes
        pred_labels = pred_labels_new
        
    return pred_labels, iter_info