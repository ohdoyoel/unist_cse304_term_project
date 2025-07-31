import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from sklearn.preprocessing import normalize
from src.utils import compute_adaptive_similarity, scipy_sparse_to_torch_sparse

def adaptive_label_propagation(data, labels, mask, structure_similarity=None, location_similarity=None, alpha=0.6, max_iter=1000, tol=1e-6, verbose=False):
    """
    Adaptive Label Propagation 알고리즘을 수행합니다.
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        labels: 노드 레이블
        mask: 레이블이 있는 노드를 나타내는 마스크
        structure_similarity: 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        location_similarity: 위치 유사도 딕셔너리 {(u,v): similarity_score}
        alpha: 전파 강도 (기본값: 0.6)
        max_iter: 최대 반복 횟수 (기본값: 1000)
        tol: 수렴 허용 오차 (기본값: 1e-6)
        verbose: 상세 출력 여부 (기본값: False)
    
    Returns:
        pred_labels: 예측된 레이블
        last_A: 마지막 유사도 행렬
        iter_info: 반복 과정 정보
    """
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity와 location_similarity는 반드시 제공되어야 합니다.")

    unique_labels, labels_remap = torch.unique(labels[mask], return_inverse=True)
    n = labels.size(0)
    k = unique_labels.size(0)
    device = labels.device
    
    # 초기 레이블 분포 설정
    Y = torch.zeros((n, k), device=device)
    Y[mask, labels_remap] = 1  # 레이블이 있는 노드만 one-hot으로 초기화

    pred_labels = unique_labels[Y.argmax(dim=1)]
    last_A = None
    iter_info = []

    for iter_idx in range(max_iter):
        # adaptive similarity 계산
        similarity, avg_alpha, dev_alpha = compute_adaptive_similarity(
            data=data,
            structure_similarity=structure_similarity,
            location_similarity=location_similarity,
            pred_labels=pred_labels,
            mask=mask
        )
        
        iter_info.append(f"Iter {iter_idx+1} - Avg alpha: {avg_alpha}, Dev alpha: {dev_alpha}")
        if verbose:
            print(iter_info[-1])

        # similarity matrix 생성
        adj_matrix_sparse = lil_matrix((n, n))
        for (u, v), score in similarity.items():
            adj_matrix_sparse[u, v] = score
            adj_matrix_sparse[v, u] = score  # 무방향 그래프 가정
        A = scipy_sparse_to_torch_sparse(adj_matrix_sparse.tocsr()).to(device)
        last_A = A

        # Label Propagation 단계
        AY = torch.sparse.mm(A, Y)
        Y_new = alpha * AY + (1 - alpha) * Y
        Y_new = F.normalize(Y_new, p=1, dim=1)
        
        # 레이블 변화 확인
        pred_labels_new = unique_labels[Y_new.argmax(dim=1)]
        label_changes = (pred_labels_new != pred_labels).sum().item()

        diff = torch.norm(Y_new - Y, p='fro').item()
        num_unique_labels = len(torch.unique(pred_labels_new))
        iter_info.append(f"Iter {iter_idx+1}: #labels={num_unique_labels}, changes={label_changes}, diff={diff:.6f}")
        if verbose:
            print(iter_info[-1])
        
        # 수렴 조건 체크
        if diff < tol or label_changes / n < 0.01:
            iter_info.append(f"Converged at iteration {iter_idx+1}")
            if verbose:
                print(iter_info[-1])
            break
            
        Y = Y_new
        pred_labels = pred_labels_new
            
    return pred_labels, last_A, iter_info
