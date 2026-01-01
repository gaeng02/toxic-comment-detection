import os
import json

import numpy as np
import matplotlib.pyplot as plt

from src.core.data import load_dataset
from src.core.embedding import Embedder

## change
from src.experiments.ko.experiment03_good_and_bad.model import GoodBadKNNClassifier

BASE_RESULTS_DIR = os.path.join("results", "ko", "experiment03_good_and_bad")


def compute_mean_distances (
    test_vectors : np.ndarray,
    db_vectors   : np.ndarray | None,
) -> np.ndarray :
    
    n_test = test_vectors.shape[0]

    if (db_vectors is None) or (len(db_vectors) == 0) :
        return np.full((n_test,), np.nan, dtype = float)

    dists = np.linalg.norm(
        test_vectors[:, None, :] - db_vectors[None, :, :],
        axis = 2,
    )
    mean_dists = dists.mean(axis = 1)
    return mean_dists


def compute_global_means (
    test_vectors : np.ndarray,
    y_test       : np.ndarray,
    good_vectors : np.ndarray | None,
    bad_vectors  : np.ndarray | None,
) -> tuple[float, float, float] :
    """
    GOOD DB 평균 거리, BAD DB 평균 거리, 정답 DB 평균 거리(라벨에 맞는 DB) 반환.
    각각 test 전체에 대한 '전역 평균' (NaN은 무시).
    """

    mean_good = compute_mean_distances(test_vectors, good_vectors)
    mean_bad  = compute_mean_distances(test_vectors, bad_vectors)

    # GOOD/BAD 각각 전역 평균
    def finite_mean (arr : np.ndarray) -> float :
        vals = arr[np.isfinite(arr)]
        if (len(vals) == 0) :
            return float("nan")
        return float(vals.mean())

    global_good = finite_mean(mean_good)
    global_bad  = finite_mean(mean_bad)

    # 라벨에 맞는 DB 거리 평균
    # label 1 -> BAD, label 0 -> GOOD
    matched_dist_list : list[float] = []

    for d, y in zip(range(len(y_test)), y_test) :
        if (y == 1) :
            if np.isfinite(mean_bad[d]) :
                matched_dist_list.append(mean_bad[d])
        else :
            if np.isfinite(mean_good[d]) :
                matched_dist_list.append(mean_good[d])

    if len(matched_dist_list) == 0 :
        global_matched = float("nan")
    else :
        global_matched = float(np.mean(matched_dist_list))

    return global_good, global_bad, global_matched


def main () :
    
    result_num = 8
    
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")

    # 1. config 로드
    config_path = os.path.join(result_dir, "config.json")
    if (not os.path.exists(config_path)) :
        raise RuntimeError(f"config.json not found in {result_dir}")

    with open(config_path, "r", encoding = "utf-8") as f :
        config = json.load(f)

    model_name       = config.get("model_name")
    seed_size        = int(config.get("seed_size", 50))
    k                = int(config.get("k", 1))

    print(f"[INFO] model_name       :: {model_name}")
    print(f"[INFO] dataset_language :: {"ko"}")
    print(f"[INFO] seed_size        :: {seed_size}")
    print(f"[INFO] k                :: {k}")

    # 2. train / test 로드 및 임베딩
    train_path = os.path.join("dataset", "ko", "train.csv")
    test_path  = os.path.join("dataset", "ko", "test.csv")

    print(f"[INFO] Loading train dataset :: {train_path}")
    X_train, y_train = load_dataset(train_path)
    y_train = np.array(y_train, dtype = int)
    print(f"[INFO] Train size :: {len(X_train)}")

    print(f"[INFO] Loading test dataset  :: {test_path}")
    X_test, y_test = load_dataset(test_path)
    y_test = np.array(y_test, dtype = int)
    print(f"[INFO] Test size :: {len(X_test)}")

    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name = model_name, device = None)

    print("[INFO] Encoding train texts ...")
    X_train_list = list(X_train)
    train_vectors = embedder.encode(X_train_list)   # (N_train, D)

    print("[INFO] Encoding test texts ...")
    test_vectors = embedder.encode(list(X_test))    # (N_test, D)

    # 3. KNN 분류기 생성 및 seed 초기화
    clf = GoodBadKNNClassifier(
        seed_size    = seed_size,
        k            = k,
        random_state = 1,
    )

    print(f"[INFO] Initializing GOOD/BAD DB with first {seed_size} samples ...")
    start_idx = clf._init_db(X_train_list, train_vectors, y_train)
    n_samples = len(X_train_list)

    print(f"[INFO] Online training from index {start_idx} to {n_samples - 1} ...")

    # 시간축 스텝 (100 단계)
    n_steps = 100
    check_indices = set()
    for t in range(1, n_steps + 1) :
        # 전체 학습 구간(start_idx ~ n_samples-1)을 100등분
        idx = start_idx + int((n_samples - start_idx) * t / n_steps) - 1
        idx = max(start_idx, min(n_samples - 1, idx))
        check_indices.add(idx)

    check_indices = sorted(check_indices)

    step_positions : list[float] = []
    mean_good_list : list[float] = []
    mean_bad_list  : list[float] = []
    mean_match_list: list[float] = []

    # 4. online training + 중간중간 거리 평균 측정
    current_check_idx = 0
    num_checks = len(check_indices)

    for i in range(start_idx, n_samples) :
        vec   = train_vectors[i]
        text  = X_train_list[i]
        y_true = y_train[i]

        # 현재 DB 상태로 예측
        y_pred = clf._predict_vector(vec)

        # 오분류된 경우 정답 라벨 DB에 추가
        if (y_pred != y_true) :
            if (y_true == 1) :
                # BAD DB에 추가
                if clf.bad_vectors is None :
                    clf.bad_vectors = vec[None, :]
                else :
                    clf.bad_vectors = np.vstack([clf.bad_vectors, vec])
                clf.bad_texts.append(text)
            else :
                # GOOD DB에 추가
                if clf.good_vectors is None :
                    clf.good_vectors = vec[None, :]
                else :
                    clf.good_vectors = np.vstack([clf.good_vectors, vec])
                clf.good_texts.append(text)

        # 체크 포인트에 도달했는지 확인
        if (current_check_idx < num_checks) and (i == check_indices[current_check_idx]) :
            progress = (i + 1) / n_samples  # 0~1 사이 학습 진행도

            g_mean, b_mean, m_mean = compute_global_means(
                test_vectors,
                y_test,
                clf.good_vectors,
                clf.bad_vectors,
            )

            step_positions.append(progress)
            mean_good_list.append(g_mean)
            mean_bad_list.append(b_mean)
            mean_match_list.append(m_mean)

            print(
                f"[INFO] step {current_check_idx + 1}/{num_checks} "
                f"(progress={progress:.3f}) "
                f"GOOD={g_mean:.4f}, BAD={b_mean:.4f}, MATCH={m_mean:.4f}"
            )

            current_check_idx += 1

    # 5. 그래프 저장
    analytics_dir = os.path.join(result_dir, "analytics_time_series")
    os.makedirs(analytics_dir, exist_ok = True)

    # GOOD DB 평균 거리
    plt.figure()
    plt.plot(step_positions, mean_good_list, marker = "o")
    plt.xlabel("Training progress (ratio)")
    plt.ylabel("Mean distance to GOOD DB")
    plt.title("Training progress vs mean distance to GOOD DB")
    plt.grid(True)
    plt.tight_layout()
    path_good = os.path.join(analytics_dir, "time_mean_good.png")
    plt.savefig(path_good)
    plt.close()
    print(f"[INFO] Saved :: {path_good}")

    # BAD DB 평균 거리
    plt.figure()
    plt.plot(step_positions, mean_bad_list, marker = "o")
    plt.xlabel("Training progress (ratio)")
    plt.ylabel("Mean distance to BAD DB")
    plt.title("Training progress vs mean distance to BAD DB")
    plt.grid(True)
    plt.tight_layout()
    path_bad = os.path.join(analytics_dir, "time_mean_bad.png")
    plt.savefig(path_bad)
    plt.close()
    print(f"[INFO] Saved :: {path_bad}")

    # 정답 라벨에 맞는 DB 평균 거리
    plt.figure()
    plt.plot(step_positions, mean_match_list, marker = "o")
    plt.xlabel("Training progress (ratio)")
    plt.ylabel("Mean distance to matched DB")
    plt.title("Training progress vs mean distance to matched DB")
    plt.grid(True)
    plt.tight_layout()
    path_match = os.path.join(analytics_dir, "time_mean_matched.png")
    plt.savefig(path_match)
    plt.close()
    print(f"[INFO] Saved :: {path_match}")

    print("[INFO] Time-series distance analysis finished.")


if (__name__ == "__main__") :
    main()
