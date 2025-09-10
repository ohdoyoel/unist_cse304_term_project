import torch
import networkx as nx
from networkx.algorithms import community
from collections import defaultdict, Counter
from itertools import groupby
import random
from src.utils import compute_adaptive_similarity

# def asyn_lpa_communities(G, weight=None, seed=None):
#     """Returns communities in `G` as detected by asynchronous label
#     propagation.

#     The asynchronous label propagation algorithm is described in
#     [1]_. The algorithm is probabilistic and the found communities may
#     vary on different executions.

#     The algorithm proceeds as follows. After initializing each node with
#     a unique label, the algorithm repeatedly sets the label of a node to
#     be the label that appears most frequently among that nodes
#     neighbors. The algorithm halts when each node has the label that
#     appears most frequently among its neighbors. The algorithm is
#     asynchronous because each node is updated without waiting for
#     updates on the remaining nodes.

#     This generalized version of the algorithm in [1]_ accepts edge
#     weights.

#     Parameters
#     ----------
#     G : Graph

#     weight : string
#         The edge attribute representing the weight of an edge.
#         If None, each edge is assumed to have weight one. In this
#         algorithm, the weight of an edge is used in determining the
#         frequency with which a label appears among the neighbors of a
#         node: a higher weight means the label appears more often.

#     seed : integer, random_state, or None (default)
#         Indicator of random number generation state.
#         See :ref:`Randomness<randomness>`.

#     Returns
#     -------
#     communities : iterable
#         Iterable of communities given as sets of nodes.

#     Notes
#     -----
#     Edge weight attributes must be numerical.

#     References
#     ----------
#     .. [1] Raghavan, Usha Nandini, Réka Albert, and Soundar Kumara. "Near
#            linear time algorithm to detect community structures in large-scale
#            networks." Physical Review E 76.3 (2007): 036106.
#     """

#     labels = {n: i for i, n in enumerate(G)}
#     cont = True

#     while cont:
#         cont = False
#         nodes = list(G)
#         seed.shuffle(nodes)

#         for node in nodes:
#             if not G[node]:
#                 continue

#             # Get label frequencies among adjacent nodes.
#             # Depending on the order they are processed in,
#             # some nodes will be in iteration t and others in t-1,
#             # making the algorithm asynchronous.
#             if weight is None:
#                 # initialising a Counter from an iterator of labels is
#                 # faster for getting unweighted label frequencies
#                 label_freq = Counter(map(labels.get, G[node]))
#             else:
#                 # updating a defaultdict is substantially faster
#                 # for getting weighted label frequencies
#                 label_freq = defaultdict(float)
#                 for _, v, wt in G.edges(node, data=weight, default=1):
#                     label_freq[labels[v]] += wt

#             # Get the labels that appear with maximum frequency.
#             max_freq = max(label_freq.values())
#             best_labels = [
#                 label for label, freq in label_freq.items() if freq == max_freq
#             ]

#             # If the node does not have one of the maximum frequency labels,
#             # randomly choose one of them and update the node's label.
#             # Continue the iteration as long as at least one node
#             # doesn't have a maximum frequency label.
#             if labels[node] not in best_labels:
#                 labels[node] = seed.choice(best_labels)
#                 cont = True

#     yield from groups(labels).values()

plot_order = 0

def plot_graph(G, node_labels, iter_idx, target_node, structure_similarity=None, location_similarity=None, alpha_values=None, adaptive_similarity=None, all_possible_labels=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import matplotlib.colors as mcolors

    global plot_order
    plot_order += 1

    # 노드의 실제 위치 정보 가져오기
    pos = nx.get_node_attributes(G, 'pos')
    labels = {n: node_labels[n] for n in G.nodes()}
    unique_labels = sorted(set(labels.values()))
    n_labels = len(unique_labels)
    
    # 전체 가능한 라벨들을 기반으로 고정된 색상 매핑 생성
    if all_possible_labels is not None:
        # 전체 라벨 범위를 사용하여 일관된 색상 매핑
        all_labels = sorted(set(all_possible_labels))
        color_map = plt.cm.tab10 if len(all_labels) <= 10 else plt.cm.tab20
        color_mapping = {label: color_map(i / max(1, len(all_labels)-1)) for i, label in enumerate(all_labels)}
    else:
        # 현재 노드 개수를 기반으로 고정된 색상 매핑 (0부터 노드 수-1까지)
        total_nodes = len(G.nodes())
        color_map = plt.cm.tab10 if total_nodes <= 10 else plt.cm.tab20
        color_mapping = {label: color_map(label / max(1, total_nodes-1)) for label in range(total_nodes)}
    
    node_colors = [color_mapping[labels[n]] for n in G.nodes()]
    
    plt.figure(figsize=(8, 6))
    
    # 노드의 실제 좌표를 사용하여 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    
    # target_node로 수신되는 엣지들만 방향성 엣지로 표시
    target_directed_edges = []
    normal_edges = []
    
    for edge in G.edges():
        u, v = edge
        # target_node로 들어오는 엣지인지 확인 (양방향 고려)
        if v == target_node:
            target_directed_edges.append((u, v))  # u -> target_node
        elif u == target_node:
            target_directed_edges.append((v, u))  # v -> target_node
        else:
            normal_edges.append(edge)  # target_node와 관련없는 엣지
    
    # 일반 엣지 그리기 (화살표 없음)
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, alpha=0.3, width=1)
    
    # 방향성 엣지 그리기 (화살표 있음)
    if target_directed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=target_directed_edges, alpha=0.5, width=2, 
                              arrows=True, arrowsize=20, arrowstyle='->', 
                              connectionstyle='arc3,rad=0', node_size=1000)
    
    # 노드 라벨과 알파 값 표시
    node_label_dict = {}
    for node in G.nodes():
        label = labels[node]
        if alpha_values is not None and len(alpha_values) > node:
            alpha_val = alpha_values[node]
            node_label_dict[node] = f'{node}\nα:{alpha_val:.2f}'
        else:
            node_label_dict[node] = f'{node}'
    
    nx.draw_networkx_labels(G, pos, labels=node_label_dict, font_size=8, font_weight='bold')
    
    # target_node를 향한 directed_edges에 대해서만 유사도 정보 표시
    if all(sim is not None for sim in [structure_similarity, location_similarity, adaptive_similarity]) and target_directed_edges:
        for edge in target_directed_edges:
            u, v = edge
            edge_key = (v, u)

            if edge_key in structure_similarity and edge_key in location_similarity:
                # 엣지 중점 계산
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 유사도 값들 가져오기
                jaccard_sim = structure_similarity[edge_key]
                loc_sim = location_similarity[edge_key]
                
                # adaptive_similarity는 엣지별 딕셔너리
                adaptive_sim = adaptive_similarity.get(edge_key, 0.0) if adaptive_similarity else 0.0
                
                # 연결된 두 노드의 알파값 중 적절한 값 선택
                edge_alpha = 0.5  # 기본값
                selected_node_info = ""
                
                if alpha_values is not None and len(alpha_values) > max(u, v):
                    alpha_u = alpha_values[u] if u < len(alpha_values) else 0.5
                    alpha_v = alpha_values[v] if v < len(alpha_values) else 0.5
                    
                    # adaptive_similarity 계산에 더 적합한 알파값 선택
                    # adaptive_sim = alpha * jaccard_sim + (1-alpha) * loc_sim 형태라고 가정
                    # 실제 adaptive_sim에 더 가까운 결과를 주는 알파값 선택
                    if jaccard_sim != loc_sim:  # 분모가 0이 아닌 경우만
                        # adaptive_sim을 만들어내는 이론적 알파값 계산
                        theoretical_alpha = (adaptive_sim - loc_sim) / (jaccard_sim - loc_sim)
                        theoretical_alpha = max(0, min(1, theoretical_alpha))  # 0-1 범위로 제한
                        
                        # 두 노드의 알파값 중 이론적 알파값에 더 가까운 것 선택
                        if abs(alpha_u - theoretical_alpha) <= abs(alpha_v - theoretical_alpha):
                            edge_alpha = alpha_u
                            selected_node_info = f"(from {u})"
                        else:
                            edge_alpha = alpha_v
                            selected_node_info = f"(from {v})"
                    else:
                        # jaccard_sim과 loc_sim이 같은 경우, 평균값 사용
                        edge_alpha = (alpha_u + alpha_v) / 2
                        selected_node_info = "(avg)"
                
                # 엣지 정보 텍스트 생성 (선택된 노드 정보 포함)
                edge_info = f'S:{jaccard_sim:.2f}\nL:{loc_sim:.2f}\nα:{edge_alpha:.2f}\nA:{adaptive_sim:.2f}'
                
                # 텍스트 표시 (약간 오프셋을 줘서 겹치지 않게)
                offset_x = (y2 - y1) * 0.1  # 수직 오프셋
                offset_y = (x1 - x2) * 0.1  # 수평 오프셋
                
                plt.text(mid_x, mid_y, edge_info, 
                        fontsize=7, ha='center', va='center', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                        rotation=0)
    
    # 범례 생성 (현재 존재하는 라벨만 표시)
    for label in unique_labels:
        plt.scatter([], [], color=color_mapping[label], s=100, label=f'Label {label}', alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # x, y축 범위 고정 (모든 노드의 좌표를 고려)
    all_x = [pos[node][0] for node in G.nodes()]
    all_y = [pos[node][1] for node in G.nodes()]
    x_margin = (max(all_x) - min(all_x)) * 0.1  # 10% 여백
    y_margin = (max(all_y) - min(all_y)) * 0.1  # 10% 여백
    
    plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    plt.ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    plt.title(f'Adaptive Label Propagation - Iteration {iter_idx} - Node {target_node}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # 축 비율 동일하게 설정
    
    # result/custom 디렉토리 생성 (존재하지 않는 경우)
    import os
    os.makedirs('result/custom', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'result/custom/{plot_order}_alp_plot_{iter_idx}_{target_node}.png', dpi=150, bbox_inches='tight')
    plt.close()

def adaptive_label_propagation(data, labels, structure_similarity=None, location_similarity=None, max_iter=1000, verbose=False, order='random', save_plot=False):
    """
    asyn_lpa_communities 기반의 Adaptive Label Propagation 알고리즘을 수행합니다.
    매 반복마다 adaptive similarity를 다시 계산하여 가중치를 업데이트합니다.
    
    Args:
        data: Data 객체 (edge_index와 num_nodes 필요)
        labels: 노드 레이블 (초기 레이블로 사용)
        structure_similarity: 구조적 유사도 딕셔너리 {(u,v): similarity_score}
        location_similarity: 위치 유사도 딕셔너리 {(u,v): similarity_score}
        max_iter: 최대 반복 횟수 (기본값: 1000)
        verbose: 상세 출력 여부 (기본값: False)
        order: 노드 업데이트 순서 (기본값: 'random')
        save_plot: 그래프 저장 여부 (기본값: False)
    Returns:
        pred_labels: 예측된 레이블
        last_similarity: 마지막 유사도 딕셔너리
        iter_info: 반복 과정 정보
    """
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity와 location_similarity는 반드시 제공되어야 합니다.")

    n = labels.size(0)
    alpha_info = []
    iter_info = []
    last_similarity = None
    
    # NetworkX 그래프 생성 (기본 구조)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # 노드 위치 정보 설정 (data.x의 좌표 사용)
    pos_dict = {}
    if hasattr(data, 'x') and data.x is not None:
        coordinates = data.x.cpu().numpy() if hasattr(data.x, 'cpu') else data.x
        for i in range(n):
            pos_dict[i] = (coordinates[i, 0], coordinates[i, 1])  # (lon, lat) 순서로 설정
    nx.set_node_attributes(G, pos_dict, 'pos')
    
    # 기본 엣지 추가 (가중치는 매 반복마다 업데이트)
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    edges = [(int(edge_array[0, i]), int(edge_array[1, i])) for i in range(edge_array.shape[1])]
    G.add_edges_from(edges)
    
    # 초기 레이블 설정 (각 노드에 고유 레이블)
    node_labels = {n: i for i, n in enumerate(G)}
    
    if verbose:
        print(f"그래프 생성 완료: 노드 {G.number_of_nodes()}개, 엣지 {G.number_of_edges()}개")
    
    for iter_idx in range(max_iter):
        # 현재 레이블 상태로 torch tensor 생성
        current_labels = torch.tensor([node_labels[i] for i in range(n)], dtype=labels.dtype)
        
        # 적응형 유사도 계산
        adaptive_similarity, alpha, avg_alpha, dev_alpha = compute_adaptive_similarity(
            data=data,
            structure_similarity=structure_similarity,
            location_similarity=location_similarity,
            pred_labels=current_labels
        )
        # print(adaptive_similarity)
        # print(alpha)
        last_similarity = adaptive_similarity
        alpha_info.append(alpha)
        
        # 그래프 가중치 업데이트
        for u, v in G.edges():
            if (u, v) in adaptive_similarity:
                G[u][v]['weight'] = adaptive_similarity[(u, v)]
            elif (v, u) in adaptive_similarity:
                G[v][u]['weight'] = adaptive_similarity[(v, u)]
            else:
                G[u][v]['weight'] = 0.0
                G[v][u]['weight'] = 0.0
        
        if verbose:
            print(f"Iter {iter_idx+1} - Avg alpha: {avg_alpha:.6f}, Dev alpha: {dev_alpha:.6f}")
        
        # asyn_lpa_communities 로직 적용 (한 번의 전체 노드 업데이트)
        cont = False
        nodes = list(G)
        if order == 'random':
            random.shuffle(nodes)
        elif order == 'original':
            pass
        elif order == 'reverse':
            nodes.reverse()
        
        label_changes = 0

        all_nodes = list(range(len(G.nodes())))
        
        for node in nodes:
            if not G[node]:  # 이웃이 없는 노드는 건너뜀
                continue

            if save_plot:
                plot_graph(G, node_labels, iter_idx+1, node,
                        structure_similarity=structure_similarity, 
                        location_similarity=location_similarity, 
                        alpha_values=alpha, 
                        adaptive_similarity=adaptive_similarity,
                        all_possible_labels=all_nodes)
            
            # 이웃들의 레이블 빈도 계산 (가중치 고려)
            label_freq = defaultdict(float)
            for _, v, wt in G.edges(node, data='weight', default=0.0):
                if wt > 0:  # 양수 가중치만 고려
                    label_freq[node_labels[v]] += wt
            
            if not label_freq:  # 유효한 이웃이 없는 경우
                continue
                
            # 최대 빈도를 가진 레이블들 찾기
            max_freq = max(label_freq.values())
            best_labels = [
                label for label, freq in label_freq.items() if freq == max_freq
            ]
            
            # 현재 레이블이 최대 빈도 레이블 중 하나가 아니면 업데이트
            if node_labels[node] not in best_labels:
                old_label = node_labels[node]
                node_labels[node] = random.choice(best_labels)
                if old_label != node_labels[node]:
                    label_changes += 1
                    cont = True
        
        
        # 현재 상태 정보 수집
        unique_labels = len(set(node_labels.values()))
        label_change_rate = label_changes / n
        
        iter_info.append(f"Iter {iter_idx+1}: #labels={unique_labels}, changes={label_changes}({label_change_rate:.4f}), avg_alpha={avg_alpha:.6f}, dev_alpha={dev_alpha:.6f}")
        
        if verbose:
            print(iter_info[-1])
        
        # 개선된 수렴 조건들
        convergence_conditions = []
        
        # 1. 레이블 변화가 없으면 즉시 수렴
        if not cont or label_changes == 0:
            convergence_conditions.append("no_label_changes")
        
        # 2. 레이블 변화율이 매우 낮으면 수렴 (1% 미만)
        elif label_change_rate < 0.015:
            convergence_conditions.append("low_change_rate")
        
        # 수렴 조건 중 하나라도 만족하면 종료
        if convergence_conditions:
            reason = ", ".join(convergence_conditions)
            iter_info.append(f"Converged at iteration {iter_idx+1} (reason: {reason})")
            if verbose:
                print(iter_info[-1])
            break
            
    # 최종 결과를 torch tensor로 변환
    pred_labels = torch.tensor([node_labels[i] for i in range(n)], dtype=labels.dtype)
    
    # 레이블을 0부터 시작하도록 재매핑
    unique_labels = torch.unique(pred_labels)
    label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    for i in range(n):
        pred_labels[i] = label_mapping[pred_labels[i].item()]
    
    return pred_labels, last_similarity, iter_info, alpha_info
