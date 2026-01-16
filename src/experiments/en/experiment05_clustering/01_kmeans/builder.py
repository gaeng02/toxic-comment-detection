import os
import time

import numpy as np
from sklearn.cluster import KMeans

from src.core.data import load_dataset
from src.core.embedding import Embedder
from src.core.metrics import write_time, write_config


# ===== config =====
k = 500

dataset = "en"

method = f"01_kmeans/{k}"
TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")

METHOD_DIR = os.path.join("results", dataset, "experiment05_clustering", method)
os.makedirs(METHOD_DIR, exist_ok = True)

GOOD_DB_PATH = os.path.join(METHOD_DIR, "good_db_vectors.npy")
BAD_DB_PATH  = os.path.join(METHOD_DIR, "bad_db_vectors.npy")
CONFIG_PATH  = os.path.join(METHOD_DIR, "config.json")
TIME_PATH    = os.path.join(METHOD_DIR, "time.txt")


def run_kmeans_per_class (
    vectors : np.ndarray,
    n_clusters : int,
    label_name : str,
) -> np.ndarray :
    
    n_samples = vectors.shape[0]

    if (n_samples == 0) :
        print(f"[WARN] No samples for class = {label_name}. Return empty array.")
        return np.empty((0, vectors.shape[1]), dtype = np.float32)
    
    n_clusters_eff = min(n_clusters, n_samples)
    if (n_clusters_eff <= 0) :
        print(f"[WARN] Effective cluster number <= 0 for class = {label_name}. Return empty array.")
        return np.empty((0, vectors.shape[1]), dtype = np.float32)
    
    print(f"[INFO] Running KMeans :: class = {label_name}, "
          f"n_samples = {n_samples}, n_clusters = {n_clusters_eff}")
    
    kmeans = KMeans(
        n_clusters = n_clusters_eff,
        random_state = 1,
        # n_init = 10,
    )
    kmeans.fit(vectors)

    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"[INFO] KMeans finished :: class = {label_name}, centers shape = {centers.shape}")
    
    return centers


def main () :
    
    # 1. 데이터 로드
    print(f"[INFO] Loading train dataset from {TRAIN_PATH} ...")
    X_train, y_train = load_dataset(TRAIN_PATH)
    print(f"[INFO] Train size :: {len(X_train)}")

    y_train = np.asarray(y_train, dtype = int)

    # 2. 임베딩 모델 로드
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(
        model_name = model_name,
        device = None,
    )
    print("[INFO] Embedding model loaded.")

    # 3. 전체 train 임베딩
    print("[INFO] Encoding train texts ...")
    t0 = time.perf_counter()
    train_vectors = embedder.encode(list(X_train))  # (N, D)
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Train embedding")
    print(f"[INFO] Train embedding done :: shape = {train_vectors.shape}")

    # 4. 클래스별 분리 (0 = good, 1 = bad)
    is_bad  = (y_train == 1)
    is_good = (y_train == 0)

    good_vectors_all = train_vectors[is_good]
    bad_vectors_all  = train_vectors[is_bad]

    print(f"[INFO] Good samples :: {good_vectors_all.shape[0]}")
    print(f"[INFO] Bad  samples :: {bad_vectors_all.shape[0]}")

    # 5. 각 클래스별 KMeans
    t0 = time.perf_counter()
    good_db_vectors = run_kmeans_per_class(
        vectors    = good_vectors_all,
        n_clusters = k,
        label_name = "good",
    )
    bad_db_vectors = run_kmeans_per_class(
        vectors    = bad_vectors_all,
        n_clusters = k,
        label_name = "bad",
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "KMeans clustering")

    # 6. 결과 저장 (.npy)
    np.save(GOOD_DB_PATH, good_db_vectors)
    np.save(BAD_DB_PATH,  bad_db_vectors)
    print(f"[INFO] Saved GOOD DB vectors :: {GOOD_DB_PATH}")
    print(f"[INFO] Saved BAD  DB vectors :: {BAD_DB_PATH}")

    # 7. config 저장
    config = {
        "dataset_language" : dataset,
        "method"           : method,
        "embedder_model"   : model_name,
        "k"                : k,
    }
    write_config(config, CONFIG_PATH)
    print(f"[INFO] Saved config :: {CONFIG_PATH}")

    print("[INFO] 01_kmeans DB build finished.")


if (__name__ == "__main__") :
    main()
