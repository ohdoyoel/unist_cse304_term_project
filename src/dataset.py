import torch
import numpy as np
import os
import pandas as pd

def load_brightkite(sample_size=None):
    # Brightkite_edges.txt 로딩
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_path = os.path.join(base_dir, '..', 'data', 'Brightkite_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # 체크인 데이터 로딩 및 최근 방문 장소 추출
    checkin_path = os.path.join(base_dir, '..', 'data', 'Brightkite_totalCheckins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'loc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # 비정상적인 위치 데이터 제외
    # NaN 값 제외
    df = df.dropna(subset=['lat', 'lon'])
    # 위도: -90° ~ 90°, 경도: -180° ~ 180°
    df = df[
        (df['lat'] != 0) | (df['lon'] != 0)  # (0, 0) 위치 제외
    ]
    # 비정상 위치 데이터 필터링
    df = df[
        (df['lat'] >= -90) & (df['lat'] <= 90) &  # 위도 범위 체크
        (df['lon'] >= -180) & (df['lon'] <= 180)  # 경도 범위 체크
    ]
    
    recent = df.groupby('user').last().reset_index()
    
    # 최근 방문 장소가 있는 노드만 선택
    valid_nodes = set(recent['user'].astype(int))
    
    # 샘플링 적용
    if sample_size is not None and sample_size < len(valid_nodes):
        # 재현성을 위한 랜덤 시드 설정
        np.random.seed(42)
        # 무작위로 sample_size만큼의 노드 선택
        valid_nodes = set(np.random.choice(list(valid_nodes), size=sample_size, replace=False))
    
    # 엣지 필터링 - 양쪽 노드 모두 valid한 경우만 유지
    valid_edges = [(u, v) for u, v in zip(edges[0], edges[1]) 
                   if u in valid_nodes and v in valid_nodes]
    if not valid_edges:
        raise ValueError("No valid edges found after filtering")
    
    # 노드 ID 리매핑
    node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
    edges_remapped = np.array([[node_map[u], node_map[v]] for u, v in valid_edges]).T
    num_nodes = len(node_map)
    
    # feature matrix 생성
    features = np.zeros((num_nodes, 2), dtype=np.float32)
    rad_features = np.zeros((num_nodes, 2), dtype=np.float32)
    longitude = np.zeros(num_nodes, dtype=np.float32)
    latitude = np.zeros(num_nodes, dtype=np.float32)
    
    for _, row in recent.iterrows():
        old_id = int(row['user'])
        if old_id in node_map:
            new_id = node_map[old_id]
            features[new_id] = [row['lat'], row['lon']]
            rad_features[new_id] = [np.radians(row['lat']), np.radians(row['lon'])]
            longitude[new_id] = row['lon']
            latitude[new_id] = row['lat']

    # Data 객체 생성
    data = type('Data', (), {})()
    data.edge_index = torch.tensor(edges_remapped, dtype=torch.long)
    data.num_nodes = num_nodes
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.x = torch.from_numpy(features)
    data.rad_x = torch.from_numpy(rad_features)
    data.longitude = torch.from_numpy(longitude)
    data.latitude = torch.from_numpy(latitude)

    # 노드 ID 매핑 정보도 저장
    data.node_map = node_map
    
    return data, 1

def load_gowalla(sample_size=None):
    # Gowalla_edges.txt 로딩
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_path = os.path.join(base_dir, '..', 'data', 'Gowalla_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # 체크인 데이터 로딩 및 최근 방문 장소 추출
    checkin_path = os.path.join(base_dir, '..', 'data', 'Gowalla_totalCheckins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'loc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # 비정상적인 위치 데이터 제외
    # NaN 값 제외
    df = df.dropna(subset=['lat', 'lon'])
    
    # 위도: -90° ~ 90°, 경도: -180° ~ 180°
    df = df[
        (df['lat'] != 0) | (df['lon'] != 0)  # (0, 0) 위치 제외
    ]
    # 비정상 위치 데이터 필터링
    df = df[
        (df['lat'] >= -90) & (df['lat'] <= 90) &  # 위도 범위 체크
        (df['lon'] >= -180) & (df['lon'] <= 180)  # 경도 범위 체크
    ]
    
    recent = df.groupby('user').last().reset_index()
    
    # 최근 방문 장소가 있는 노드만 선택
    valid_nodes = set(recent['user'].astype(int))
    
    # 샘플링 적용
    if sample_size is not None and sample_size < len(valid_nodes):
        # 재현성을 위한 랜덤 시드 설정
        np.random.seed(42)
        # 무작위로 sample_size만큼의 노드 선택
        valid_nodes = set(np.random.choice(list(valid_nodes), size=sample_size, replace=False))
    
    # 엣지 필터링 - 양쪽 노드 모두 valid한 경우만 유지
    valid_edges = [(u, v) for u, v in zip(edges[0], edges[1]) 
                   if u in valid_nodes and v in valid_nodes]
    if not valid_edges:
        raise ValueError("No valid edges found after filtering")
    
    # 노드 ID 리매핑
    node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
    edges_remapped = np.array([[node_map[u], node_map[v]] for u, v in valid_edges]).T
    num_nodes = len(node_map)
    
    # feature matrix 생성
    features = np.zeros((num_nodes, 2), dtype=np.float32)
    rad_features = np.zeros((num_nodes, 2), dtype=np.float32)
    longitude = np.zeros(num_nodes, dtype=np.float32)
    latitude = np.zeros(num_nodes, dtype=np.float32)
    
    for _, row in recent.iterrows():
        old_id = int(row['user'])
        if old_id in node_map:
            new_id = node_map[old_id]
            features[new_id] = [row['lat'], row['lon']]
            rad_features[new_id] = [np.radians(row['lat']), np.radians(row['lon'])]
            longitude[new_id] = row['lon']
            latitude[new_id] = row['lat']

    # Data 객체 생성
    data = type('Data', (), {})()
    data.edge_index = torch.tensor(edges_remapped, dtype=torch.long)
    data.num_nodes = num_nodes
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.x = torch.from_numpy(features)
    data.rad_x = torch.from_numpy(rad_features)
    data.longitude = torch.from_numpy(longitude)
    data.latitude = torch.from_numpy(latitude)
    
    # 노드 ID 매핑 정보도 저장
    data.node_map = node_map
    
    return data, 1

def load_dataset(name='brightkite', sample_size=None):
    if name.lower() == 'brightkite':
        return load_brightkite(sample_size)
    if name.lower() == 'gowalla':
        return load_gowalla(sample_size)
    else:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        dataset = Planetoid(root=os.path.join('../data', name), name=name, transform=NormalizeFeatures())
        data = dataset[0]
        print(data, dataset.num_classes)
        return data, dataset.num_classes