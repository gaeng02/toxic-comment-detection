import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked

from src.core.data import load_dataset
from src.core.embedding import Embedder

## change
BASE_RESULTS_DIR = os.path.join("results", "en", "experiment02_threshold")
TEST_PATH = os.path.join("dataset", "en", "test.csv")


def find_close_cases(
    result_num: int = 2,
    distance_threshold: float = 0.1,
    top_n : int = 50
):
    """
    유클리드 거리가 threshold 이하인 test-DB 쌍들을 찾아서 출력
    
    Args:
        result_num: 결과 디렉토리 번호
        distance_threshold: 이 거리 이하인 케이스들을 찾음 (default=0.1)
        top_n: 출력할 최대 케이스 수
    """
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")
    
    # Config 로드
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"config.json not found in {result_dir}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_name = config.get("model_name")
    
    # DB 벡터 및 텍스트 로드
    bad_vec_path = os.path.join(result_dir, "bad_db_vectors.npy")
    good_vec_path = os.path.join(result_dir, "good_db_vectors.npy")
    bad_csv_path = os.path.join(result_dir, "bad_db.csv")
    good_csv_path = os.path.join(result_dir, "good_db.csv")
    
    bad_vectors = np.load(bad_vec_path) if os.path.exists(bad_vec_path) else None
    good_vectors = np.load(good_vec_path) if os.path.exists(good_vec_path) else None
    
    bad_db_df = pd.read_csv(bad_csv_path, encoding="utf-8-sig") if os.path.exists(bad_csv_path) else None
    good_db_df = pd.read_csv(good_csv_path, encoding="utf-8-sig") if os.path.exists(good_csv_path) else None
    
    print(f"[INFO] BAD  DB vectors: {0 if bad_vectors is None else len(bad_vectors)}")
    print(f"[INFO] GOOD DB vectors: {0 if good_vectors is None else len(good_vectors)}")
    
    # Test 데이터 로드
    print(f"[INFO] Loading test dataset :: {TEST_PATH}")
    X_test, y_test = load_dataset(TEST_PATH)
    X_test = list(X_test)
    y_test = np.array(y_test, dtype=int)
    print(f"[INFO] Test size :: {len(X_test)}")
    
    # Embedding 모델 로드
    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name=model_name, device=None)
    print(f"[INFO] Encoding test texts :: {len(X_test)}")
    test_vectors = embedder.encode(X_test)
    
    # 거리가 threshold 이하인 케이스 찾기
    close_cases = []
    
    # BAD DB와의 거리 확인
    if bad_vectors is not None and bad_db_df is not None:
        print(f"[INFO] Finding cases with distance <= {distance_threshold} to BAD DB...")
        
        gen = pairwise_distances_chunked(
            test_vectors,
            bad_vectors,
            metric='euclidean',
            n_jobs=1,
            working_memory=1024,
        )
        
        chunk_start_idx = 0
        for dists_chunk in gen:
            for j, dists in enumerate(dists_chunk):
                test_idx = chunk_start_idx + j
                # threshold 이하인 모든 DB 인덱스 찾기
                close_db_indices = np.where(dists <= distance_threshold)[0]
                for db_idx in close_db_indices:
                    close_cases.append({
                        'test_idx': test_idx,
                        'test_text': X_test[test_idx],
                        'test_label': y_test[test_idx],
                        'db_type': 'BAD',
                        'db_idx': int(db_idx),
                        'db_text': bad_db_df.iloc[db_idx]['content'],
                        'distance': float(dists[db_idx]),
                    })
            chunk_start_idx += dists_chunk.shape[0]
    
    # GOOD DB와의 거리 확인
    if good_vectors is not None and good_db_df is not None:
        print(f"[INFO] Finding cases with distance <= {distance_threshold} to GOOD DB...")
        
        gen = pairwise_distances_chunked(
            test_vectors,
            good_vectors,
            metric='euclidean',
            n_jobs=1,
            working_memory=1024,
        )
        
        chunk_start_idx = 0
        for dists_chunk in gen:
            for j, dists in enumerate(dists_chunk):
                test_idx = chunk_start_idx + j
                # threshold 이하인 모든 DB 인덱스 찾기
                close_db_indices = np.where(dists <= distance_threshold)[0]
                for db_idx in close_db_indices:
                    close_cases.append({
                        'test_idx': test_idx,
                        'test_text': X_test[test_idx],
                        'test_label': y_test[test_idx],
                        'db_type': 'GOOD',
                        'db_idx': int(db_idx),
                        'db_text': good_db_df.iloc[db_idx]['content'],
                        'distance': float(dists[db_idx]),
                    })
            chunk_start_idx += dists_chunk.shape[0]
    
    # 거리순으로 정렬
    close_cases.sort(key=lambda x: x['distance'])
    
    # 상위 N개만 출력
    print(f"\n{'='*80}")
    print(f"[INFO] Found {len(close_cases)} cases with distance <= {distance_threshold}")
    print(f"[INFO] Showing top {min(top_n, len(close_cases))} cases:")
    print(f"{'='*80}\n")
    
    for i, case in enumerate(close_cases[:top_n], 1):
        print(f"\n[Case {i}] Distance: {case['distance']:.6f}")
        print(f"  Test Index: {case['test_idx']}")
        print(f"  Test Label: {case['test_label']} ({'BAD' if case['test_label'] == 1 else 'GOOD'})")
        print(f"  DB Type: {case['db_type']}")
        print(f"  DB Index: {case['db_idx']}")
        print(f"  Test Text: {case['test_text']}")
        print(f"  DB Text:   {case['db_text']}")
        print(f"  Same text: {'YES' if case['test_text'] == case['db_text'] else 'NO'}")
        print("-" * 80)
    
    # 결과를 파일로 저장
    output_dir = os.path.join(result_dir, "analytics_distances")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "close_cases.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Close Cases Analysis (Euclidean distance <= {distance_threshold})\n")
        f.write(f"Total cases found: {len(close_cases)}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, case in enumerate(close_cases[:top_n], 1):
            f.write(f"\n[Case {i}] Distance: {case['distance']:.6f}\n")
            f.write(f"  Test Index: {case['test_idx']}\n")
            f.write(f"  Test Label: {case['test_label']} ({'BAD' if case['test_label'] == 1 else 'GOOD'})\n")
            f.write(f"  DB Type: {case['db_type']}\n")
            f.write(f"  DB Index: {case['db_idx']}\n")
            f.write(f"  Test Text: {case['test_text']}\n")
            f.write(f"  DB Text:   {case['db_text']}\n")
            f.write(f"  Same text: {'YES' if case['test_text'] == case['db_text'] else 'NO'}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\n[INFO] Results saved to {output_path}")
    
    # 통계 출력
    if len(close_cases) > 0 :
        same_text_count = sum(1 for case in close_cases if case['test_text'] == case['db_text'])
        print(f"\n[INFO] Statistics:")
        print(f"  - Total close cases: {len(close_cases)}")
        print(f"  - Same text cases: {same_text_count} ({same_text_count/len(close_cases)*100:.1f}%)")
        
        bad_db_cases = sum(1 for case in close_cases if case['db_type'] == 'BAD')
        good_db_cases = sum(1 for case in close_cases if case['db_type'] == 'GOOD')
        print(f"  - BAD DB cases: {bad_db_cases}")
        print(f"  - GOOD DB cases: {good_db_cases}")
        
        test_label1_cases = sum(1 for case in close_cases if case['test_label'] == 1)
        test_label0_cases = sum(1 for case in close_cases if case['test_label'] == 0)
        print(f"  - Test label 1 (BAD): {test_label1_cases}")
        print(f"  - Test label 0 (GOOD): {test_label0_cases}")
    else :
        print(f"\n[INFO] No cases found with distance <= {distance_threshold}")


if (__name__ == "__main__") :

    result_num = 2
    threshold = 0.1
    top_n = 50
    
    find_close_cases(
        result_num=result_num,
        distance_threshold=threshold,
        top_n=top_n,
    )

