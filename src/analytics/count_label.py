import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_chunked

from src.core.data import load_dataset
from src.core.embedding import Embedder

BASE_RESULTS_DIR = os.path.join("results", "en", "experiment02_threshold")
TEST_PATH = os.path.join("dataset", "en", "test.csv")


def predict_label(
    vec: np.ndarray,
    good_vectors: np.ndarray | None,
    bad_vectors: np.ndarray | None,
    k: int,
    threshold_distance: float | None,
    break_label: int,
) -> int:
    """
    단일 벡터에 대한 예측 수행 (model.py의 _predict_vector와 동일한 로직)
    """
    if (good_vectors is None) and (bad_vectors is None):
        raise RuntimeError("GOOD DB와 BAD DB가 모두 비어 있습니다.")
    
    dists_all = []
    labels_all = []
    
    # good DB 거리
    if (good_vectors is not None) and (len(good_vectors) > 0):
        dists_good = np.linalg.norm(good_vectors - vec, axis=1)
        dists_all.extend(dists_good.tolist())
        labels_all.extend([0] * len(dists_good))  # good = 0
    
    # bad DB 거리
    if (bad_vectors is not None) and (len(bad_vectors) > 0):
        dists_bad = np.linalg.norm(bad_vectors - vec, axis=1)
        dists_all.extend(dists_bad.tolist())
        labels_all.extend([1] * len(dists_bad))  # bad = 1
    
    if not dists_all:
        raise RuntimeError("GOOD / BAD DB에 유효한 벡터가 없습니다.")
    
    dists_all = np.asarray(dists_all, dtype=float)
    labels_all = np.asarray(labels_all, dtype=int)
    
    k_eff = min(k, len(dists_all))
    
    # threshold 거리
    if threshold_distance is not None:
        mask = (dists_all <= threshold_distance)
        
        if np.any(mask):
            idx_candidates = np.where(mask)[0]
            
            if (k is not None) and (k > 0) and (len(idx_candidates) > k):
                idx_sorted = idx_candidates[np.argsort(dists_all[idx_candidates])]
                idx_used = idx_sorted[:k]
            else:
                idx_used = idx_candidates
            
            neighbor_labels = labels_all[idx_used]
            count_good = int(np.sum(neighbor_labels == 0))
            count_bad = int(np.sum(neighbor_labels == 1))
            
            if count_bad > count_good:
                return 1
            if count_good > count_bad:
                return 0
            
            return int(break_label)
    
    # fallback
    idx = np.argpartition(dists_all, k_eff - 1)[:k_eff]
    neighbor_labels = labels_all[idx]
    
    # 다수결 (0 = good, 1 = bad), 동률이면 bad(1) 쪽으로
    mean_label = neighbor_labels.mean()
    return 1 if mean_label >= 0.5 else 0


def main () :
    
    result_num = 5
    
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")

    config_path = os.path.join(result_dir, "config.json")
    if (not os.path.exists(config_path)) :
        raise RuntimeError(f"config.json not found in {result_dir}")

    with open(config_path, "r", encoding = "utf-8") as f :
        config = json.load(f)

    model_name = config.get("model_name")
    threshold_distance = config.get("threshold_distance")
    k = config.get("k")
    break_label = config.get("break_label", 1)

    if (threshold_distance is None) :
        raise ValueError("threshold_distance가 config에 없습니다.")
    if (k is None) :
        raise ValueError("k가 config에 없습니다.")

    print(f"[INFO] threshold_distance :: {threshold_distance}")
    print(f"[INFO] k :: {k}")

    bad_vec_path = os.path.join(result_dir, "bad_db_vectors.npy")
    good_vec_path = os.path.join(result_dir, "good_db_vectors.npy")

    bad_vectors = np.load(bad_vec_path) if os.path.exists(bad_vec_path) else None
    good_vectors = np.load(good_vec_path) if os.path.exists(good_vec_path) else None

    print(f"[INFO] BAD  DB vectors: {0 if bad_vectors is None else len(bad_vectors)}")
    print(f"[INFO] GOOD DB vectors: {0 if good_vectors is None else len(good_vectors)}")

    print(f"[INFO] Loading test dataset :: {TEST_PATH}")
    X_test, y_test = load_dataset(TEST_PATH)
    X_test = list(X_test)
    y_test = np.array(y_test, dtype = int)
    print(f"[INFO] Test size :: {len(X_test)}")

    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name = model_name, device = None)
    print(f"[INFO] Encoding test texts :: {len(X_test)}")
    test_vectors = embedder.encode(X_test)

    print("[INFO] Computing label counts within threshold for each test sample...")

    # GOOD DB와의 거리 계산
    good_counts = []
    if (good_vectors is not None) and (len(good_vectors) > 0) :
        print(f"[INFO] Computing distances to GOOD DB...")
        gen_good = pairwise_distances_chunked(
            test_vectors,
            good_vectors,
            metric = 'euclidean',
            n_jobs = 1,
            working_memory = 1024,
        )
        
        chunk_start_idx = 0
        for dists_chunk in gen_good :
            for j, dists in enumerate(dists_chunk) :
                # threshold 이하인 neighbor 개수 (label 0 = good)
                count_good = int(np.sum(dists <= threshold_distance))
                good_counts.append(count_good)
            chunk_start_idx += dists_chunk.shape[0]
    else :
        good_counts = [0] * len(X_test)

    # BAD DB와의 거리 계산
    bad_counts = []
    if (bad_vectors is not None) and (len(bad_vectors) > 0) :
        print(f"[INFO] Computing distances to BAD DB...")
        gen_bad = pairwise_distances_chunked(
            test_vectors,
            bad_vectors,
            metric = 'euclidean',
            n_jobs = 1,
            working_memory = 1024,
        )
        
        chunk_start_idx = 0
        for dists_chunk in gen_bad :
            for j, dists in enumerate(dists_chunk) :
                # threshold 이하인 neighbor 개수 (label 1 = bad)
                count_bad = int(np.sum(dists <= threshold_distance))
                bad_counts.append(count_bad)
            chunk_start_idx += dists_chunk.shape[0]
    else :
        bad_counts = [0] * len(X_test)

    # 예측 라벨 계산
    print("[INFO] Computing predicted labels...")
    y_pred = []
    for vec in test_vectors:
        pred = predict_label(
            vec,
            good_vectors,
            bad_vectors,
            k=k,
            threshold_distance=threshold_distance,
            break_label=break_label,
        )
        y_pred.append(pred)
    y_pred = np.array(y_pred, dtype=int)

    # 결과 데이터프레임 생성
    df_results = pd.DataFrame({
        "test_index": range(len(X_test)),
        "test_label": y_test,
        "predicted_label": y_pred,
        "count_label_0": good_counts,  # GOOD DB (label 0)
        "count_label_1": bad_counts,  # BAD DB (label 1)
        "count_total": [g + b for g, b in zip(good_counts, bad_counts)],
    })

    # CSV 저장
    analytics_dir = os.path.join(result_dir, "analytics_distances")
    os.makedirs(analytics_dir, exist_ok = True)

    output_path = os.path.join(analytics_dir, "label_counts.csv")
    df_results.to_csv(output_path, index = False, encoding = "utf-8-sig")
    print(f"[INFO] Results saved to {output_path}")

    # 통계 출력
    print(f"\n[INFO] Statistics:")
    print(f"  - Total test samples: {len(df_results)}")
    print(f"  - Samples with neighbors in threshold: {np.sum(df_results['count_total'] > 0)}")
    print(f"  - Samples with no neighbors in threshold: {np.sum(df_results['count_total'] == 0)}")
    print(f"  - Average count_label_0: {df_results['count_label_0'].mean():.2f}")
    print(f"  - Average count_label_1: {df_results['count_label_1'].mean():.2f}")
    print(f"  - Average count_total: {df_results['count_total'].mean():.2f}")

    # Label별 통계
    print(f"\n[INFO] Statistics by test label:")
    for label_value, label_name in [(1, "label1"), (0, "label0")] :
        mask = (y_test == label_value)
        label_df = df_results[mask]
        print(f"  - {label_name} ({'BAD' if label_value == 1 else 'GOOD'}):")
        print(f"    - Count: {len(label_df)}")
        print(f"    - Avg count_label_0: {label_df['count_label_0'].mean():.2f}")
        print(f"    - Avg count_label_1: {label_df['count_label_1'].mean():.2f}")
        print(f"    - Avg count_total: {label_df['count_total'].mean():.2f}")

    print("[INFO] Label count analysis finished.")


if (__name__ == "__main__") :
    main()

