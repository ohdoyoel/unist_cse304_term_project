import torch

def louvain_method(adj_matrix):
    import numpy as np
    import networkx as nx
    import scipy.sparse

    # adj_matrix가 torch sparse tensor라면 scipy sparse로 변환
    if hasattr(adj_matrix, 'is_sparse') and adj_matrix.is_sparse:
        if adj_matrix.layout == torch.sparse_csr:
            # CSR format
            indices = adj_matrix.col_indices().cpu().numpy()
            indptr = adj_matrix.crow_indices().cpu().numpy()
            data = adj_matrix.values().cpu().numpy()
            adj_matrix = scipy.sparse.csr_matrix(
                (data, indices, indptr),
                shape=adj_matrix.shape
            )
        else:
            # COO format
            coo = adj_matrix.cpu().coalesce()
            indices = coo.indices().numpy()
            data = coo.values().numpy()
            adj_matrix = scipy.sparse.coo_matrix(
                (data, (indices[0], indices[1])),
                shape=adj_matrix.shape
            )

    G = nx.from_scipy_sparse_array(adj_matrix)
    try:
        import community as community_louvain
    except ImportError:
        import community.community_louvain as community_louvain
    partition = community_louvain.best_partition(G)
    return np.array(list(partition.values()))