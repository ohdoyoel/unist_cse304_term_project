import torch
from collections import defaultdict
from src.utils import compute_fixed_alpha_similarity

def fixed_alpha_label_propagation(data, labels, fixed_alpha, structure_similarity=None, location_similarity=None, max_iter=1000, verbose=False):
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity와 location_similarity는 반드시 제공되어야 합니다.")

    n = labels.size(0)
    prev_unique_labels = n
    prev_changes = n
    
    # 초기 레이블 설정
    pred_labels = labels.clone()

    iter_info = []
    last_adj_dict = None

    # 인접 리스트 생성
    # edge_index로부터 인접 리스트 생성
    adj_list = [[] for _ in range(n)]
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    for i in range(edge_array.shape[1]):
        u, v = edge_array[0, i], edge_array[1, i]
        adj_list[u].append(v)
        adj_list[v].append(u)  # 무방향 그래프 가정
    adj_weights = [defaultdict(float) for _ in range(n)]

    for iter_idx in range(max_iter):
        # 적응형 유사도 계산
        similarity, avg_alpha, dev_alpha = compute_fixed_alpha_similarity(
            data=data,
            fixed_alpha=fixed_alpha,
            structure_similarity=structure_similarity,
            location_similarity=location_similarity,
            pred_labels=pred_labels
        )
        last_adj_dict = similarity

        # 인접 리스트와 가중치 업데이트
        for i in range(n):
            adj_weights[i].clear()
        
        for (u, v), score in similarity.items():
            adj_weights[u][v] = score

        # 가중치 기반 레이블 전파
        pred_labels_new = pred_labels.clone()
        label_changes = 0

        for node in range(n):
            if not adj_list[node]:  # 이웃이 없는 노드는 건너뜀
                continue

            # 이웃들의 레이블과 가중치 수집
            neighbor_labels = defaultdict(float)
            total_weight = 0
            
            for neighbor in adj_list[node]:
                weight = adj_weights[node][neighbor]
                neighbor_labels[pred_labels[neighbor].item()] += weight
                total_weight += weight

            if total_weight > 0:  # 정규화
                for label in neighbor_labels:
                    neighbor_labels[label] /= total_weight

            # 가장 높은 가중치를 가진 레이블 선택
            if neighbor_labels:
                # 최대 가중치 찾기
                max_weight = max(neighbor_labels.values())
                # 최대 가중치를 가진 레이블들 찾기
                best_labels = [label for label, weight in neighbor_labels.items() if weight == max_weight]
                
                # tie가 발생한 경우 랜덤으로 선택
                if len(best_labels) > 1:
                    best_label = best_labels[torch.randint(0, len(best_labels), (1,)).item()]
                else:
                    best_label = best_labels[0]
                
                if pred_labels[node].item() != best_label:
                    pred_labels_new[node] = best_label
                    label_changes += 1

        num_unique_labels = len(torch.unique(pred_labels_new))
        iter_info.append(f"Iter {iter_idx+1}: #labels={num_unique_labels}, changes={label_changes}")
        if verbose:
            print(iter_info[-1])

        # 수렴 조건 체크
        if (
            label_changes / n < 0.01  # 충분히 안정화됨
            or (num_unique_labels >= prev_unique_labels and label_changes >= prev_changes)  # 레이블 수가 더 이상 줄지 않음
        ):
            iter_info.append(f"Converged at iteration {iter_idx+1}")
            if verbose:
                print(iter_info[-1])
            break

        prev_unique_labels = num_unique_labels
        prev_changes = label_changes
        pred_labels = pred_labels_new

    return pred_labels, last_adj_dict, iter_info
