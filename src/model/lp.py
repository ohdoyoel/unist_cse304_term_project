import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def label_propagation(A, labels, mask, alpha=0.6, max_iter=10, tol=1e-6, adaptive_similarity_fn=None, data=None, features=None, verbose=False):
    """
    If adaptive_similarity_fn is provided, A will be recomputed at each iteration using the function:
    adaptive_similarity_fn(data, features, pred_labels)
    """
    unique_labels, labels_remap = torch.unique(labels[mask], return_inverse=True)
    n = labels.size(0)
    k = unique_labels.size(0)
    # Data 객체가 들어오면 edge_index 등에서 adj_matrix를 직접 생성
    if hasattr(A, "edge_index") and hasattr(A, "num_nodes"):
        # A가 Data 객체라면, edge_index로부터 sparse adj 생성
        from scipy.sparse import coo_matrix
        edge_array = A.edge_index.cpu().numpy() if hasattr(A.edge_index, 'cpu') else A.edge_index
        adj_matrix_sparse = coo_matrix(
            (np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
            shape=(A.num_nodes, A.num_nodes)
        )
        from src.utils import scipy_sparse_to_torch_sparse
        A = scipy_sparse_to_torch_sparse(adj_matrix_sparse).to(labels.device)
    # device 결정: A가 None이거나 device 속성이 없으면 labels.device 사용
    if A is not None and hasattr(A, "device"):
        device = A.device
    else:
        device = labels.device
    Y = torch.zeros((n, k), device=device)
    Y[mask, labels_remap] = 1  # Only labeled nodes are one-hot

    pred_labels = unique_labels[Y.argmax(dim=1)]
    last_A = A
    # print(adaptive_similarity_fn, data, features)
    for iter_idx in range(max_iter):
        # adaptive similarity 업데이트
        if adaptive_similarity_fn is not None and data is not None and features is not None:
            similarity, avg_alpha = adaptive_similarity_fn(data, features=features, pred_labels=pred_labels)
            print(f"Iter {iter_idx+1} - Avg alpha: {avg_alpha}")
            from scipy.sparse import lil_matrix
            adj_matrix_sparse = lil_matrix((n, n))
            for (u, v), score in similarity.items():
                adj_matrix_sparse[u, v] = score
                adj_matrix_sparse[v, u] = score
            from src.utils import scipy_sparse_to_torch_sparse
            A = scipy_sparse_to_torch_sparse(adj_matrix_sparse.tocsr()).to(device)
            last_A = A

        if A is not None and hasattr(A, "is_sparse") and A.is_sparse:
            AY = torch.sparse.mm(A, Y)
        else:
            AY = torch.mm(A, Y)
        Y_new = alpha * AY + (1 - alpha) * Y
        if torch.norm(Y_new - Y, p='fro') < tol:
            break
        Y = Y_new
        pred_labels = unique_labels[Y.argmax(dim=1)]
        if verbose:
            print(f"Iter {iter_idx+1}: #labels={len(torch.unique(pred_labels))}")
    return unique_labels[Y.argmax(dim=1)], last_A
