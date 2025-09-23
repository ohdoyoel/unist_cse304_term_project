import pandas as pd
import numpy as np
import os
from sklearn.metrics import silhouette_score
import warnings

# 경고 메시지 무시 (너무 많은 데이터로 인한 성능 경고 등)
warnings.filterwarnings('ignore')

def calculate_silhouette_score(csv_file_path):
    """
    CSV 파일에서 실루엣 점수를 계산하는 함수
    
    Args:
        csv_file_path (str): nodes.csv 파일 경로
    
    Returns:
        float: 실루엣 점수 (실패시 None)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)
        
        # 좌표 데이터 추출 (longitude, latitude)
        X = df[['longitude', 'latitude']].values
        
        # 클러스터 라벨 추출
        labels = df['cluster_label'].values
        
        # 클러스터가 1개만 있는 경우 실루엣 점수 계산 불가
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"경고: 클러스터가 1개만 있음 - {csv_file_path}")
            return None
        
        # 실루엣 점수 계산 (유클리디안 거리 사용)
        score = silhouette_score(X, labels, metric='euclidean')
        
        return score
    
    except Exception as e:
        print(f"에러 발생 - 파일: {csv_file_path}, 에러: {e}")
        return None

def analyze_dataset_algorithm_silhouette(dataset, algorithm):
    """
    특정 데이터셋과 알고리즘에 대해 50개 실험의 실루엣 점수 분석
    
    Args:
        dataset (str): 데이터셋 이름 (brightkite, gowalla)
        algorithm (str): 알고리즘 이름 (lp, jlp, jllp, llp, alp)
    
    Returns:
        tuple: (평균, 표준편차, 전체 통계 정보)
    """
    base_path = f"result/{dataset}/{algorithm}"
    
    # 50개 실험의 실루엣 점수들을 저장할 리스트
    silhouette_scores = []
    
    print(f"\n=== {dataset.upper()} - {algorithm.upper()} 실루엣 점수 분석 중... ===")
    
    for i in range(50):
        # 각 실험의 nodes.csv 파일 경로
        csv_file = f"{base_path}/{dataset}_{algorithm}_{i}_nodes.csv"
        
        if os.path.exists(csv_file):
            # 실루엣 점수 계산
            score = calculate_silhouette_score(csv_file)
            
            if score is not None:
                silhouette_scores.append(score)
                print(f"실험 {i:2d}: 실루엣 점수 = {score:8.4f}")
            else:
                print(f"실험 {i:2d}: 점수 계산 실패")
        else:
            print(f"실험 {i:2d}: 파일 없음 - {csv_file}")
    
    if silhouette_scores:
        # 50개 실험의 실루엣 점수들에 대한 평균과 표준편차 계산
        overall_mean = np.mean(silhouette_scores)
        overall_std = np.std(silhouette_scores, ddof=1)  # 표본 표준편차
        
        print(f"\n📊 최종 결과:")
        print(f"   50개 실험의 실루엣 점수 평균: {overall_mean:.6f}")
        print(f"   50개 실험의 실루엣 점수 표준편차: {overall_std:.6f}")
        print(f"   성공한 실험 개수: {len(silhouette_scores)}/50")
        
        return overall_mean, overall_std, {
            'experiment_count': len(silhouette_scores),
            'scores': silhouette_scores,
            'min_score': min(silhouette_scores),
            'max_score': max(silhouette_scores)
        }
    else:
        print("❌ 분석할 데이터가 없습니다.")
        return None, None, None

def main():
    """
    메인 함수 - 모든 데이터셋과 알고리즘 조합에 대해 실루엣 점수 분석 수행
    """
    # 분석할 데이터셋과 알고리즘 목록
    datasets = ['brightkite', 'gowalla']
    algorithms = ['lp', 'jlp', 'jllp', 'llp', 'alp']
    
    # 결과 저장용 딕셔너리
    results = {}
    
    print("🚀 실루엣 점수 분석을 시작합니다!")
    print("=" * 60)
    print("📝 참고: 실루엣 점수는 -1에서 1 사이의 값으로,")
    print("   1에 가까울수록 클러스터링이 잘 된 것을 의미합니다.")
    print("=" * 60)
    
    # 모든 조합에 대해 분석 수행
    for dataset in datasets:
        results[dataset] = {}
        
        for algorithm in algorithms:
            # 각 데이터셋-알고리즘 조합 분석
            mean, std, stats = analyze_dataset_algorithm_silhouette(dataset, algorithm)
            
            results[dataset][algorithm] = {
                'mean': mean,
                'std': std,
                'stats': stats
            }
    
    # 최종 요약 결과 출력
    print("\n" + "=" * 80)
    print("🎯 최종 실루엣 점수 요약 결과")
    print("=" * 80)
    
    print(f"{'Dataset':<12} {'Algorithm':<8} {'Mean':<12} {'Std':<12} {'성공 실험':<10}")
    print("-" * 70)
    
    for dataset in datasets:
        for algorithm in algorithms:
            result = results[dataset][algorithm]
            if result['mean'] is not None:
                success_count = result['stats']['experiment_count']
                print(f"{dataset:<12} {algorithm:<8} {result['mean']:<12.6f} {result['std']:<12.6f} {success_count}/50")
            else:
                print(f"{dataset:<12} {algorithm:<8} {'N/A':<12} {'N/A':<12} {'0/50':<10}")
    
    print("\n📈 실루엣 점수 해석:")
    print("   • 0.7 ~ 1.0  : 강한 클러스터 구조")
    print("   • 0.5 ~ 0.7  : 적당한 클러스터 구조") 
    print("   • 0.25 ~ 0.5 : 약한 클러스터 구조")
    print("   • 0.0 ~ 0.25 : 겹치는 클러스터")
    print("   • -1.0 ~ 0.0 : 잘못된 클러스터링")
    print("\n✅ 분석 완료!")

if __name__ == "__main__":
    main()
