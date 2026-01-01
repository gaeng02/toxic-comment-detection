# src/experiments/en/analytics_threshold_distance.py
# 혹은 ko 쪽이면 경로만 ko로 바꿔서 사용

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_chunked

from src.core.data import load_dataset
from src.core.embedding import Embedder
from src.experiments.en.experiment02_threshold.model import GoodBadDistanceClassifier
# ↑ 경로/파일명은 네 실제 구조에 맞게 수정해줘


# ===== 실험 설정 =====
dataset = "en"  # "ko"로 바꿔서 써도 됨

TEST_PATH = os.path.join("dataset", dataset, "test.csv")

BASE_RESULTS_DIR = os.path.join(
    "results",
    dataset,
    "experiment02_threshold",
)

# 분석하고 싶은 결과 번호 (예: 01, 02, 08 ...)
result_num = 6
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]  # 보고 싶은 threshold 값들


def compute_min_distances(
    test_vectors: np.ndarray,
    good_vectors: np.ndarray | None,
    bad_vectors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_test = test_vectors.shape[0]
    d_min_good = np.full(n_test, np.inf, dtype=float)
    d_min_bad = np.full(n_test, np.inf, dtype=float)

    # 메모리 효율적인 배치 처리 - GOOD
    if good_vectors is not None and len(good_vectors) > 0:
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
                d_min_good[test_idx] = np.min(dists)
            chunk_start_idx += dists_chunk.shape[0]

    # 메모리 효율적인 배치 처리 - BAD
    if bad_vectors is not None and len(bad_vectors) > 0:
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
                d_min_bad[test_idx] = np.min(dists)
            chunk_start_idx += dists_chunk.shape[0]

    d_min_all = np.minimum(d_min_good, d_min_bad)
    return d_min_good, d_min_bad, d_min_all


def plot_min_distance_histogram(
    d_min_all: np.ndarray,
    out_dir: str,
    fname: str = "hist_min_distance_all.png",
):
    vals = d_min_all[np.isfinite(d_min_all)]
    if len(vals) == 0:
        print("[WARN] No finite distances to plot.")
        return

    plt.figure()
    plt.hist(vals, bins=50)
    plt.xlabel("Min distance to ANY DB sample")
    plt.ylabel("Count")
    plt.title("Histogram of min distance (good/bad DB combined)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved histogram :: {out_path}")


def main() :
    # ================== 0. 결과 폴더, config, 벡터 로드 ==================
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")

    if not os.path.isdir(result_dir):
        raise RuntimeError(f"Result directory not found: {result_dir}")

    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"config.json not found in {result_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name")
    seed_size = config.get("seed_size", 20)
    k = config.get("k", 1)
    threshold_init = config.get("threshold_distance", 5.0)
    break_label = config.get("break_label", 1)

    print(f"[INFO] model_name        :: {model_name}")
    print(f"[INFO] seed_size         :: {seed_size}")
    print(f"[INFO] k                 :: {k}")
    print(f"[INFO] threshold_init    :: {threshold_init}")
    print(f"[INFO] break_label       :: {break_label}")

    bad_vec_path = os.path.join(result_dir, "bad_db_vectors.npy")
    good_vec_path = os.path.join(result_dir, "good_db_vectors.npy")

    if not os.path.exists(bad_vec_path) and not os.path.exists(good_vec_path):
        raise RuntimeError("No GOOD/BAD DB vectors found in result_dir.")

    bad_vectors = np.load(bad_vec_path) if os.path.exists(bad_vec_path) else None
    good_vectors = np.load(good_vec_path) if os.path.exists(good_vec_path) else None

    print(f"[INFO] BAD  DB vectors: {0 if bad_vectors is None else len(bad_vectors)}")
    print(f"[INFO] GOOD DB vectors: {0 if good_vectors is None else len(good_vectors)}")

    analytics_dir = os.path.join(result_dir, "analytics_threshold")
    os.makedirs(analytics_dir, exist_ok=True)

    # ================== 1. test 셋 로드 + 임베딩 ==================
    print(f"[INFO] Loading test dataset :: {TEST_PATH}")
    X_test, y_test = load_dataset(TEST_PATH)
    y_test = np.array(y_test, dtype=int)
    print(f"[INFO] Test size :: {len(X_test)}")

    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name=model_name, device=None)

    print(f"[INFO] Encoding test texts :: {len(X_test)}")
    test_vectors = embedder.encode(list(X_test))

    # ================== 2. 거리 통계 계산 ==================
    print("[INFO] Computing min distances to GOOD / BAD DB ...")
    d_min_good, d_min_bad, d_min_all = compute_min_distances(
        test_vectors, good_vectors, bad_vectors
    )

    # raw 저장
    np.save(os.path.join(analytics_dir, "d_min_good.npy"), d_min_good)
    np.save(os.path.join(analytics_dir, "d_min_bad.npy"), d_min_bad)
    np.save(os.path.join(analytics_dir, "d_min_all.npy"), d_min_all)
    np.save(os.path.join(analytics_dir, "y_test.npy"), y_test)
    print(f"[INFO] Saved raw distances :: {analytics_dir}")

    # 2-1. 전체 min distance 히스토그램
    plot_min_distance_histogram(d_min_all, analytics_dir)

    # ================== 3. threshold 별 coverage 계산 ==================
    coverage_stats: list[tuple[float, float]] = []

    for thr in THRESHOLDS:
        has_neighbor = (d_min_all <= thr)
        ratio = float(has_neighbor.mean())
        coverage_stats.append((thr, ratio))
        print(f"[INFO] Threshold {thr:.4f} -> coverage = {ratio * 100:.2f}%")

    coverage_path = os.path.join(analytics_dir, "threshold_coverage.txt")
    with open(coverage_path, "w", encoding="utf-8") as f:
        for thr, ratio in coverage_stats:
            f.write(f"threshold={thr:.4f}, coverage={ratio:.6f}\n")
    print(f"[INFO] Saved threshold coverage :: {coverage_path}")

    # ================== 4. threshold 값에 따른 예측 변화량 ==================
    # 이미 학습된 DB를 재사용하고, threshold_distance만 바꿔가며 predict
    preds_by_thr: dict[float, np.ndarray] = {}

    # 공통 classifier 인스턴스 (DB는 직접 세팅)
    clf = GoodBadDistanceClassifier(
        seed_size=seed_size,
        k=k,
        threshold_distance=threshold_init,
        break_label=break_label,
        random_state=config.get("random_state", 1),
    )

    clf.good_vectors = good_vectors
    clf.bad_vectors = bad_vectors

    for thr in THRESHOLDS :
        clf.threshold_distance = thr
        print(f"[INFO] Predicting with threshold = {thr:.4f}")
        y_pred = clf.predict(X_test, embedder=embedder)
        preds_by_thr[thr] = y_pred

        acc = float((y_pred == y_test).mean())
        print(f"       -> accuracy = {acc * 100:.2f}%")

    # 기준 threshold는 리스트의 첫 번째 값
    base_thr = THRESHOLDS[0]
    base_pred = preds_by_thr[base_thr]

    diffs_path = os.path.join(analytics_dir, "threshold_diffs.txt")
    with open(diffs_path, "w", encoding="utf-8") as f:
        f.write(f"Base threshold = {base_thr:.4f}\n\n")
        for thr in THRESHOLDS[1:]:
            y_pred = preds_by_thr[thr]
            n_diff = int((y_pred != base_pred).sum())
            f.write(f"threshold={thr:.4f}, diff_count={n_diff}\n")
            print(f"[INFO] Compare thr={base_thr:.4f} vs thr={thr:.4f} -> diff={n_diff}")

    print(f"[INFO] Saved threshold diff stats :: {diffs_path}")
    print("[INFO] Threshold sensitivity analysis finished.")


if __name__ == "__main__":
    main()
