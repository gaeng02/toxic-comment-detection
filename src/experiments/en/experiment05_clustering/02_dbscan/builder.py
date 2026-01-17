import os
import time

import numpy as np
from sklearn.cluster import DBSCAN

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.core.metrics import write_time, write_config


# ===== config =====
dataset = "en"

# DBSCAN hyper-parameters
eps_good = 1.0
eps_bad  = 1.0

min_samples_good = 5
min_samples_bad  = 5


method = "02_dbscan"
TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")

RESULTS_DIR = os.path.join("results", dataset, "experiment05_clustering", method)

def compute_medoid (vectors : np.ndarray) -> np.ndarray :
    """
    클러스터 벡터들 (n_cluster, D)에서
    centroid에 가장 가까운 실제 샘플(= medoid)을 반환.
    """
    if vectors.shape[0] == 1 :
        return vectors[0]

    centroid = vectors.mean(axis = 0, keepdims = True)   # (1, D)
    dists = np.linalg.norm(vectors - centroid, axis = 1) # (n_cluster,)
    idx = int(np.argmin(dists))
    return vectors[idx]


def run_dbscan_per_class (
    vectors : np.ndarray,
    labels : np.ndarray,
    target_label : int,
    eps : float,
    min_samples : int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
    """
    특정 클래스(target_label)에 대해 DBSCAN 수행 후
    - 각 클러스터는 medoid + 해당 클러스터 size
    - noise(-1)는 각 샘플 그대로 prototype + count = 1

    return:
        proto_vectors : (M, D)
        proto_labels  : (M,)   # 모두 target_label
        proto_counts  : (M,)   # prototype이 대표하는 샘플 개수
    """
    mask = (labels == target_label)
    class_vectors = vectors[mask]

    print(f"[INFO] Class = {target_label} | n_samples = {class_vectors.shape[0]}")

    if class_vectors.shape[0] == 0 :
        print(f"[WARN] No samples for class = {target_label}.")
        return (
            np.empty((0, vectors.shape[1]), dtype = np.float32),
            np.empty((0,), dtype = int),
            np.empty((0,), dtype = int),
        )

    print(f"[INFO] Running DBSCAN :: label = {target_label}, "
          f"eps = {eps}, min_samples = {min_samples}")

    dbscan = DBSCAN(
        eps = eps,
        min_samples = min_samples,
    )
    cluster_labels = dbscan.fit_predict(class_vectors)  # (n_samples_class,)

    unique_clusters = np.unique(cluster_labels)
    print(f"[INFO] Found clusters for label = {target_label} :: {unique_clusters.tolist()}")

    proto_vec_list   : list[np.ndarray] = []
    proto_label_list : list[int] = []
    proto_count_list : list[int] = []

    for cid in unique_clusters :
        cid_mask = (cluster_labels == cid)
        cluster_vecs = class_vectors[cid_mask]   # (n_cluster_i, D)

        if cid == -1 :
            # noise: 각 점을 그대로 prototype으로 (count = 1)
            n_noise = cluster_vecs.shape[0]
            print(f"[INFO] label = {target_label}, noise samples = {n_noise}")
            for v in cluster_vecs :
                proto_vec_list.append(v)
                proto_label_list.append(target_label)
                proto_count_list.append(1)
        else :
            # 정상 클러스터: medoid + cluster_size
            n_cluster = cluster_vecs.shape[0]
            medoid = compute_medoid(cluster_vecs)

            proto_vec_list.append(medoid)
            proto_label_list.append(target_label)
            proto_count_list.append(n_cluster)

            print(f"[INFO] label = {target_label}, cluster_id = {cid}, size = {n_cluster}")

    if not proto_vec_list :
        return (
            np.empty((0, vectors.shape[1]), dtype = np.float32),
            np.empty((0,), dtype = int),
            np.empty((0,), dtype = int),
        )

    proto_vectors = np.stack(proto_vec_list, axis = 0).astype(np.float32)
    proto_labels  = np.asarray(proto_label_list, dtype = int)
    proto_counts  = np.asarray(proto_count_list, dtype = int)

    print(f"[INFO] label = {target_label} | prototypes = {proto_vectors.shape[0]}")
    return proto_vectors, proto_labels, proto_counts


def main () :
    
    numbering = find_numbering(RESULTS_DIR)
    result_path = os.path.join(RESULTS_DIR, numbering)
    os.makedirs(result_path, exist_ok = True)
    print(f"[INFO] Result path :: {result_path}")

    # 통합 DB (medoid + count)
    DB_VECTORS_PATH = os.path.join(result_path, "db_vectors.npy")
    DB_LABELS_PATH  = os.path.join(result_path, "db_labels.npy")
    DB_COUNTS_PATH  = os.path.join(result_path, "db_counts.npy")

    # 기존 코드 호환용 good / bad 분리 벡터
    GOOD_DB_PATH = os.path.join(result_path, "good_db_vectors.npy")
    BAD_DB_PATH  = os.path.join(result_path, "bad_db_vectors.npy")

    CONFIG_PATH = os.path.join(result_path, "config.json")
    TIME_PATH   = os.path.join(result_path, "time.txt")


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

    # 4. 클래스별 DBSCAN + medoid + count (0 = good, 1 = bad)
    print("[INFO] Running DBSCAN for GOOD (label = 0) ...")
    good_vecs, good_labels, good_counts = run_dbscan_per_class(
        vectors      = train_vectors,
        labels       = y_train,
        target_label = 0,
        eps          = eps_good,
        min_samples  = min_samples_good,
    )

    print("[INFO] Running DBSCAN for BAD (label = 1) ...")
    bad_vecs, bad_labels, bad_counts = run_dbscan_per_class(
        vectors      = train_vectors,
        labels       = y_train,
        target_label = 1,
        eps          = eps_bad,
        min_samples  = min_samples_bad,
    )

    # 5. 하나의 DB로 합치기
    db_vectors = np.concatenate([good_vecs, bad_vecs], axis = 0)
    db_labels  = np.concatenate([good_labels, bad_labels], axis = 0)
    db_counts  = np.concatenate([good_counts, bad_counts], axis = 0)

    print(f"[INFO] Final DB size :: {db_vectors.shape[0]} (dim = {db_vectors.shape[1]})")
    print(f"[INFO] Label distribution in DB :: "
          f"good = {np.sum(db_labels == 0)}, bad = {np.sum(db_labels == 1)}")

    # 6. 통합 DB 저장
    np.save(DB_VECTORS_PATH, db_vectors)
    np.save(DB_LABELS_PATH,  db_labels)
    np.save(DB_COUNTS_PATH,  db_counts)
    print(f"[INFO] Saved DB vectors :: {DB_VECTORS_PATH}")
    print(f"[INFO] Saved DB labels  :: {DB_LABELS_PATH}")
    print(f"[INFO] Saved DB counts  :: {DB_COUNTS_PATH}")

    # 7. 기존 kmeans 기반 DistanceNN 코드 호환용 good / bad 분리 벡터
    good_db_vectors = db_vectors[db_labels == 0]
    bad_db_vectors  = db_vectors[db_labels == 1]

    np.save(GOOD_DB_PATH, good_db_vectors)
    np.save(BAD_DB_PATH,  bad_db_vectors)
    print(f"[INFO] Saved GOOD DB vectors :: {GOOD_DB_PATH}")
    print(f"[INFO] Saved BAD  DB vectors :: {BAD_DB_PATH}")

    # 8. config 저장
    config = {
        "dataset_language" : dataset,
        "method"           : method,
        "embedder_model"   : model_name,
        "eps_good"         : eps_good,
        "eps_bad"          : eps_bad,
        "min_samples_good" : min_samples_good,
        "min_samples_bad"  : min_samples_bad,
        "db_size"          : int(db_vectors.shape[0]),
        "db_dim"           : int(db_vectors.shape[1]),
    }
    write_config(config, CONFIG_PATH)
    print(f"[INFO] Saved config :: {CONFIG_PATH}")

    print("[INFO] 02_dbscan DB build finished.")


if (__name__ == "__main__") :
    main()
