import pandas as pd
import numpy as np
import os
from collections import Counter
import glob

def calculate_cluster_sizes(csv_file_path):
    """
    CSV 파일에서 클러스터 크기들을 계산하는 함수
    
    Args:
        csv_file_path (str): nodes.csv 파일 경로
    
    Returns:
        list: 각 클러스터의 크기 리스트
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)
        
        # cluster_label 컬럼에서 각 클러스터별 노드 개수 계산
        cluster_counts = df['cluster_label'].value_counts()
        
        # 클러스터 크기들을 리스트로 반환
        return cluster_counts.tolist()
    
    except Exception as e:
        print(f"에러 발생 - 파일: {csv_file_path}, 에러: {e}")
        return []

def analyze_dataset_algorithm(dataset, algorithm):
    """
    특정 데이터셋과 알고리즘에 대해 50개 실험의 클러스터 크기 분석
    
    Args:
        dataset (str): 데이터셋 이름 (brightkite, gowalla)
        algorithm (str): 알고리즘 이름 (lp, jlp, jllp, llp, alp)
    
    Returns:
        tuple: (평균들의 평균, 평균들의 표준편차, 전체 통계 정보)
    """
    base_path = f"result/{dataset}/{algorithm}"
    
    # 50개 실험의 평균 클러스터 크기들을 저장할 리스트
    experiment_means = []
    
    print(f"\n=== {dataset.upper()} - {algorithm.upper()} 분석 중... ===")
    
    for i in range(50):
        # 각 실험의 nodes.csv 파일 경로
        csv_file = f"{base_path}/{dataset}_{algorithm}_{i}_nodes.csv"
        
        if os.path.exists(csv_file):
            # 클러스터 크기들 계산
            cluster_sizes = calculate_cluster_sizes(csv_file)
            
            if cluster_sizes:
                # 이 실험의 평균 클러스터 크기 계산
                mean_size = np.mean(cluster_sizes)
                experiment_means.append(mean_size)
                
                print(f"실험 {i:2d}: 클러스터 개수 = {len(cluster_sizes):4d}, 평균 크기 = {mean_size:8.2f}")
            else:
                print(f"실험 {i:2d}: 데이터 없음")
        else:
            print(f"실험 {i:2d}: 파일 없음 - {csv_file}")
    
    if experiment_means:
        # 50개 실험의 평균값들에 대한 평균과 표준편차 계산
        overall_mean = np.mean(experiment_means)
        overall_std = np.std(experiment_means, ddof=1)  # 표본 표준편차
        
        print(f"\n📊 최종 결과:")
        print(f"   50개 실험의 평균 클러스터 크기들의 평균: {overall_mean:.4f}")
        print(f"   50개 실험의 평균 클러스터 크기들의 표준편차: {overall_std:.4f}")
        
        return overall_mean, overall_std, {
            'experiment_count': len(experiment_means),
            'experiment_means': experiment_means,
            'min_mean': min(experiment_means),
            'max_mean': max(experiment_means)
        }
    else:
        print("❌ 분석할 데이터가 없습니다.")
        return None, None, None

def main():
    """
    메인 함수 - 모든 데이터셋과 알고리즘 조합에 대해 분석 수행
    """
    # 분석할 데이터셋과 알고리즘 목록
    datasets = ['brightkite', 'gowalla']
    algorithms = ['lp', 'jlp', 'jllp', 'llp', 'alp']
    
    # 결과 저장용 딕셔너리
    results = {}
    
    print("🚀 클러스터 크기 분석을 시작합니다!")
    print("=" * 60)
    
    # 모든 조합에 대해 분석 수행
    for dataset in datasets:
        results[dataset] = {}
        
        for algorithm in algorithms:
            # 각 데이터셋-알고리즘 조합 분석
            mean, std, stats = analyze_dataset_algorithm(dataset, algorithm)
            
            results[dataset][algorithm] = {
                'mean': mean,
                'std': std,
                'stats': stats
            }
    
    # 최종 요약 결과 출력
    print("\n" + "=" * 80)
    print("🎯 최종 요약 결과")
    print("=" * 80)
    
    print(f"{'Dataset':<12} {'Algorithm':<8} {'Mean':<12} {'Std':<12} {'실험 개수':<8}")
    print("-" * 60)
    
    for dataset in datasets:
        for algorithm in algorithms:
            result = results[dataset][algorithm]
            if result['mean'] is not None:
                print(f"{dataset:<12} {algorithm:<8} {result['mean']:<12.4f} {result['std']:<12.4f} {result['stats']['experiment_count']:<8}")
            else:
                print(f"{dataset:<12} {algorithm:<8} {'N/A':<12} {'N/A':<12} {'0':<8}")
    
    print("\n✅ 분석 완료!")

if __name__ == "__main__":
    main()
