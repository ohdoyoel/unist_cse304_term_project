import torch
import numpy as np
import os
import pandas as pd
from collections import deque, defaultdict

def find_connected_components(edges, num_nodes, directed=False):
    """
    그래프의 연결된 구성요소를 찾는 함수
    
    Args:
        edges: 엣지 리스트 (2 x num_edges 형태)
        num_nodes: 노드 개수
        directed: 방향 그래프 여부 (False: WCC, True: SCC)
    
    Returns:
        components: 각 구성요소별 노드 리스트
        largest_component: 가장 큰 구성요소의 노드와 엣지 정보
    """
    # 인접 리스트 생성
    graph = defaultdict(list)
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        graph[u].append(v)
        if not directed:  # WCC의 경우 양방향으로 추가
            graph[v].append(u)
    
    visited = set()
    components = []
    
    for node in range(num_nodes):
        if node not in visited:
            # BFS로 연결된 구성요소 찾기
            component = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    # 가장 큰 구성요소 찾기
    largest_component = max(components, key=len)
    largest_nodes = set(largest_component)
    
    # 가장 큰 구성요소 내의 엣지 개수 계산
    largest_edges = 0
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        if u in largest_nodes and v in largest_nodes:
            largest_edges += 1
    
    return components, {
        'nodes': len(largest_component),
        'edges': largest_edges,
        'fraction_nodes': len(largest_component) / num_nodes,
        'fraction_edges': largest_edges / edges.shape[1] if edges.shape[1] > 0 else 0
    }

def find_strongly_connected_components(edges, num_nodes):
    """
    코사라주 알고리즘을 사용하여 강연결 구성요소(SCC)를 찾는 함수 (반복문 기반)
    
    Args:
        edges: 엣지 리스트 (2 x num_edges 형태)
        num_nodes: 노드 개수
    
    Returns:
        components: 각 SCC별 노드 리스트
        largest_component: 가장 큰 SCC의 노드와 엣지 정보
    """
    # 원본 그래프와 역방향 그래프 생성
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        graph[u].append(v)
        reverse_graph[v].append(u)
    
    # 1단계: 원본 그래프에서 DFS 완료 순서 기록 (반복문 기반)
    visited = set()
    finish_order = []
    
    for start_node in range(num_nodes):
        if start_node not in visited:
            # 스택을 사용한 반복적 DFS
            stack = [(start_node, False)]  # (노드, 완료 여부)
            
            while stack:
                node, finished = stack.pop()
                
                if finished:
                    # 노드 탐색 완료
                    finish_order.append(node)
                else:
                    if node not in visited:
                        visited.add(node)
                        stack.append((node, True))  # 완료 표시를 위해 다시 추가
                        
                        # 인접 노드들을 스택에 추가
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                stack.append((neighbor, False))
    
    # 2단계: 역방향 그래프에서 완료 순서의 역순으로 DFS (반복문 기반)
    visited = set()
    components = []
    
    for start_node in reversed(finish_order):
        if start_node not in visited:
            # 스택을 사용한 반복적 DFS
            component = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    
                    # 인접 노드들을 스택에 추가
                    for neighbor in reverse_graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            if component:
                components.append(component)
    
    # 가장 큰 SCC 찾기
    if not components:
        # SCC가 없는 경우 (빈 그래프)
        return [], {
            'nodes': 0,
            'edges': 0,
            'fraction_nodes': 0,
            'fraction_edges': 0
        }
    
    largest_component = max(components, key=len)
    largest_nodes = set(largest_component)
    
    # 가장 큰 SCC 내의 엣지 개수 계산
    largest_edges = 0
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        if u in largest_nodes and v in largest_nodes:
            largest_edges += 1
    
    return components, {
        'nodes': len(largest_component),
        'edges': largest_edges,
        'fraction_nodes': len(largest_component) / num_nodes,
        'fraction_edges': largest_edges / edges.shape[1] if edges.shape[1] > 0 else 0
    }

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

    # WCC와 SCC 정보 계산
    _, wcc_info = find_connected_components(edges_remapped, num_nodes, directed=False)
    _, scc_info = find_strongly_connected_components(edges_remapped, num_nodes)
    
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
    
    # WCC와 SCC 정보 추가
    data.largest_wcc_nodes = wcc_info['nodes']
    data.largest_wcc_edges = wcc_info['edges']
    data.largest_wcc_nodes_fraction = wcc_info['fraction_nodes']
    data.largest_wcc_edges_fraction = wcc_info['fraction_edges']
    
    data.largest_scc_nodes = scc_info['nodes']
    data.largest_scc_edges = scc_info['edges']
    data.largest_scc_nodes_fraction = scc_info['fraction_nodes']
    data.largest_scc_edges_fraction = scc_info['fraction_edges']
    
    # 평균 차수 추가
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
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

    # WCC와 SCC 정보 계산
    _, wcc_info = find_connected_components(edges_remapped, num_nodes, directed=False)
    _, scc_info = find_strongly_connected_components(edges_remapped, num_nodes)
    
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
    
    # WCC와 SCC 정보 추가
    data.largest_wcc_nodes = wcc_info['nodes']
    data.largest_wcc_edges = wcc_info['edges']
    data.largest_wcc_nodes_fraction = wcc_info['fraction_nodes']
    data.largest_wcc_edges_fraction = wcc_info['fraction_edges']
    
    data.largest_scc_nodes = scc_info['nodes']
    data.largest_scc_edges = scc_info['edges']
    data.largest_scc_nodes_fraction = scc_info['fraction_nodes']
    data.largest_scc_edges_fraction = scc_info['fraction_edges']
    
    # 평균 차수 추가
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
    return data, 1

def load_yelp(sample_size=None):
    # 기본 디렉토리 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 소셜 관계(엣지) 데이터 로딩
    edge_path = os.path.join(base_dir, '..', 'data', 'yelp', 'Yelp_social_relations.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # POI 좌표 데이터 로딩
    poi_path = os.path.join(base_dir, '..', 'data', 'yelp', 'Yelp_poi_coos.txt')
    poi_df = pd.read_csv(poi_path, sep='\t', header=None, names=['poi_id', 'lat', 'lon'])
    
    # 체크인 데이터 로딩 및 최근 방문 장소 추출
    checkin_path = os.path.join(base_dir, '..', 'data', 'yelp', 'Yelp_check_ins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'poi', 'time'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # 최근 체크인 정보와 POI 좌표 정보 병합
    recent = df.groupby('user').last().reset_index()
    recent = recent.merge(poi_df, left_on='poi', right_on='poi_id', how='left')
    
    # 비정상적인 위치 데이터 제외
    # NaN 값 제외
    recent = recent.dropna(subset=['lat', 'lon'])
    
    # 위도: -90° ~ 90°, 경도: -180° ~ 180°
    recent = recent[
        (recent['lat'] != 0) | (recent['lon'] != 0)  # (0, 0) 위치 제외
    ]
    # 비정상 위치 데이터 필터링
    recent = recent[
        (recent['lat'] >= -90) & (recent['lat'] <= 90) &  # 위도 범위 체크
        (recent['lon'] >= -180) & (recent['lon'] <= 180)  # 경도 범위 체크
    ]
    
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

    # WCC와 SCC 정보 계산
    _, wcc_info = find_connected_components(edges_remapped, num_nodes, directed=False)
    _, scc_info = find_strongly_connected_components(edges_remapped, num_nodes)
    
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
    
    # WCC와 SCC 정보 추가
    data.largest_wcc_nodes = wcc_info['nodes']
    data.largest_wcc_edges = wcc_info['edges']
    data.largest_wcc_nodes_fraction = wcc_info['fraction_nodes']
    data.largest_wcc_edges_fraction = wcc_info['fraction_edges']
    
    data.largest_scc_nodes = scc_info['nodes']
    data.largest_scc_edges = scc_info['edges']
    data.largest_scc_nodes_fraction = scc_info['fraction_nodes']
    data.largest_scc_edges_fraction = scc_info['fraction_edges']
    
    # 평균 차수 추가
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
    return data, 1

def load_dataset(name='brightkite', sample_size=None):
    if name.lower() == 'brightkite':
        return load_brightkite(sample_size)
    if name.lower() == 'gowalla':
        return load_gowalla(sample_size)
    if name.lower() == 'yelp':
        return load_yelp(sample_size)
    else:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        dataset = Planetoid(root=os.path.join('../data', name), name=name, transform=NormalizeFeatures())
        data = dataset[0]
        print(data, dataset.num_classes)
        return data, dataset.num_classes