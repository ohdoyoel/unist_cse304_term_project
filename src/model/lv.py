def louvain_method(adj_matrix):
    import numpy as np
    import networkx as nx
    import scipy.sparse

    # adj_matrix가 torch sparse tensor라면 scipy sparse로 변환
    if hasattr(adj_matrix, 'to_dense'):
        # torch sparse tensor -> dense -> numpy -> scipy coo_matrix
        dense = adj_matrix.to_dense().cpu().numpy()
        adj_matrix = scipy.sparse.coo_matrix(dense)

    G = nx.from_scipy_sparse_array(adj_matrix)
    try:
        import community as community_louvain
    except ImportError:
        import community.community_louvain as community_louvain
    partition = community_louvain.best_partition(G)
    return np.array(list(partition.values()))