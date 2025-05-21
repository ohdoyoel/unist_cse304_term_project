import torch
import numpy as np
import os
import pandas as pd

def load_brightkite():
    # Brightkite_edges.txt: 각 줄이 "u v" 형태의 엣지
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_path = os.path.join(base_dir, '..', 'data', 'Brightkite_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    num_nodes = edges.max() + 1
    edge_index = torch.tensor(edges, dtype=torch.long)

    # --- 가장 최근 방문 장소의 (위도, 경도) feature 생성 ---
    checkin_path = os.path.join(base_dir, '..', 'data', 'Brightkite_totalCheckins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'loc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    recent = df.groupby('user').last().reset_index()
    features = np.full((num_nodes, 2), np.nan, dtype=np.float32)
    longitude = np.full(num_nodes, np.nan, dtype=np.float32)
    latitude = np.full(num_nodes, np.nan, dtype=np.float32)
    for _, row in recent.iterrows():
        uid = int(row['user'])
        if 0 <= uid < num_nodes:
            features[uid] = [row['lat'], row['lon']]
            latitude[uid] = row['lat']
            longitude[uid] = row['lon']
    x = torch.from_numpy(features)

    # dummy labels (비지도)
    y = torch.zeros(num_nodes, dtype=torch.long)
    data = type('Data', (), {})()
    data.edge_index = edge_index
    data.num_nodes = num_nodes
    data.y = y
    data.x = x
    data.latitude = torch.from_numpy(latitude)
    data.longitude = torch.from_numpy(longitude)
    return data, 1

def load_dataset(name='brightkite'):
    if name.lower() == 'brightkite':
        return load_brightkite()
    else:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        dataset = Planetoid(root=os.path.join('../data', name), name=name, transform=NormalizeFeatures())
        data = dataset[0]
        print(data, dataset.num_classes)
        return data, dataset.num_classes