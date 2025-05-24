import torch
import numpy as np
import pandas as pd
from src.dataset import load_dataset
from src.model.gcn import GCN
from src.utils import save_graph_result, evaluate_and_save_results
from src.metrics import modularity, conductance, avg_clustering_coefficient, normalized_cut, silhouette_coefficient, graph_density
import torch.nn.functional as F

def get_long_lat(data, num_nodes):
    longitude = (
        data.longitude.cpu().numpy() if hasattr(data, 'longitude') and hasattr(data.longitude, 'cpu')
        else data.longitude if hasattr(data, 'longitude')
        else np.full(num_nodes, np.nan)
    )
    latitude = (
        data.latitude.cpu().numpy() if hasattr(data, 'latitude') and hasattr(data.latitude, 'cpu')
        else data.latitude if hasattr(data, 'latitude')
        else np.full(num_nodes, np.nan)
    )
    return longitude, latitude

def save_result(data, pred_labels, file):
    num_nodes = data.num_nodes
    longitude, latitude = get_long_lat(data, num_nodes)
    nodes_df = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'cluster_label': pred_labels.cpu().numpy() if hasattr(pred_labels, 'cpu') else pred_labels,
        'longitude': longitude,
        'latitude': latitude
    })
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    edges_df = pd.DataFrame({
        'source': edge_array[0],
        'target': edge_array[1]
    }).drop_duplicates()
    save_graph_result(nodes_df, edges_df, file)

if __name__ == '__main__':
    dataset_name = 'brightkite'
    data, _ = load_dataset(dataset_name)
    num_nodes = data.num_nodes

    # data.x의 NaN/inf 전처리
    x_np = data.x.cpu().numpy() if hasattr(data.x, 'cpu') else data.x
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
    data.x = torch.tensor(x_np, dtype=torch.float, device=data.x.device)

    # Unsupervised GCN 학습 (Graph AutoEncoder 방식)
    in_dim = data.x.size(1)
    hidden_dim = 64
    out_dim = 16
    model = GCN(in_dim, hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    # 인접행렬 생성 (sparse to dense)
    edge_index = data.edge_index
    adj_label = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj_label[edge_index[0], edge_index[1]] = 1
    adj_label = adj_label.float()  # float형으로 변환 추가

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("NaN or Inf detected in embeddings at epoch", epoch)
            break
        recon = torch.sigmoid(torch.matmul(z, z.t()))
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            print("NaN or Inf detected in recon at epoch", epoch)
            break
        loss = F.binary_cross_entropy(recon, adj_label)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 임베딩 추출 (학습된 모델 사용)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    # NaN이 포함된 노드(임베딩) 제거
    X = embeddings.cpu().numpy()
    nan_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[nan_mask]

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # 클러스터 개수 k를 직접 지정
    k = 500  # 원하는 클러스터 개수로 수정하세요

    pred_labels = np.full(X.shape[0], -1, dtype=int)
    if X_valid.shape[0] > 0:
        pred_labels[nan_mask] = KMeans(n_clusters=k, random_state=0).fit_predict(X_valid)
    else:
        print("Warning: All node embeddings contain NaN. No clustering performed.")
    pred_labels = torch.tensor(pred_labels)

    save_result(data, pred_labels, 'gcn')

    from src.utils import evaluate_and_save_results
    # adj_matrix=None이면 silhouette_coefficient 등에서 오류가 발생하므로, nan_mask가 모두 False면 silhouette 계산을 건너뜁니다.
    adj_matrix = None
    if nan_mask.any():
        from scipy.sparse import coo_matrix
        edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
        adj_matrix_sparse = coo_matrix(
            (np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
            shape=(num_nodes, num_nodes)
        )
        from src.utils import scipy_sparse_to_torch_sparse
        adj_matrix = scipy_sparse_to_torch_sparse(adj_matrix_sparse)
    else:
        print("Warning: All node embeddings contain NaN. Silhouette score will be skipped.")

    evaluate_and_save_results(
        data.edge_index, pred_labels, adj_matrix,
        "gcn_result.txt",
        "GCN + KMeans:"
    )
