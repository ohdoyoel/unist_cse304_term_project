import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize
from src.utils import scipy_sparse_to_torch_sparse

def label_propagation(similarity, labels, mask, alpha=1, max_iter=1000, tol=1e-6, verbose=True):
    """
    기본적인 Label Propagation 알고리즘을 수행합니다.
    
    Args:
        similarity: 노드 간 유사도를 담은 딕셔너리 {(u,v): similarity_score}
        labels: 노드 레이블
        mask: 레이블이 있는 노드를 나타내는 마스크
        alpha: 전파 강도 (기본값: 1)
        max_iter: 최대 반복 횟수 (기본값: 1000)
        tol: 수렴 허용 오차 (기본값: 1e-6)
        verbose: 상세 출력 여부 (기본값: True)
    
    Returns:
        pred_labels: 예측된 레이블
        iter_info: 반복 과정 정보
    """
    unique_labels, labels_remap = torch.unique(labels[mask], return_inverse=True)
    n = labels.size(0)
    k = unique_labels.size(0)
    device = labels.device

    # similarity로부터 sparse adjacency matrix 생성
    adj_matrix_sparse = lil_matrix((n, n))
    for (u, v), score in similarity.items():
        adj_matrix_sparse[u, v] = score
        adj_matrix_sparse[v, u] = score  # 무방향 그래프 가정
    
    # sparse matrix를 PyTorch sparse tensor로 변환
    A = scipy_sparse_to_torch_sparse(adj_matrix_sparse.tocsr()).to(device)
    
    # 초기 레이블 분포 설정
    Y = torch.zeros((n, k), device=device)
    Y[mask, labels_remap] = 1  # 레이블이 있는 노드만 one-hot으로 초기화

    label_idx = Y.argmax(dim=1)
    pred_labels = unique_labels[label_idx]

    iter_info = []

    for iter_idx in range(max_iter):
        # Label Propagation 단계
        AY = torch.sparse.mm(A, Y)
        Y_new = alpha * AY + (1 - alpha) * Y
        
        # 각 노드의 레이블 분포를 확률 분포로 정규화
        Y_new = F.normalize(Y_new, p=1, dim=1)
        
        # 레이블 변화 확인
        label_idx_new = Y_new.argmax(dim=1)
        pred_labels_new = unique_labels[label_idx_new]
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
        
    return pred_labels, iter_info