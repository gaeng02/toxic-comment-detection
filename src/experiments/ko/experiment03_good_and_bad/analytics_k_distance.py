import os
import json

import numpy as np
import matplotlib.pyplot as plt

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder

BASE_RESULTS_DIR = os.path.join("results", "ko", "experiment03_good_and_bad")


def compute_topk_distances (
    test_vectors : np.ndarray,
    db_vectors : np.ndarray | None,
    k : int = 3,
) -> np.ndarray :

    n_test = test_vectors.shape[0]

    if (db_vectors is None) or (len(db_vectors) == 0) :
        return np.full((n_test, k), np.nan, dtype=float)

    dists = np.linalg.norm(
        test_vectors[:, None, :] - db_vectors[None, :, :],
        axis=2,
    )

    dists_sorted = np.sort(dists, axis=1)

    if (dists_sorted.shape[1] >= k) :
        return dists_sorted[:, :k]

    pad_width = k - dists_sorted.shape[1]
    pad = np.full((n_test, pad_width), np.nan, dtype = float)
    return np.concatenate([dists_sorted, pad], axis = 1)


def plot_histogram (
    distances: np.ndarray,
    label_name: str,
    db_type: str,
    rank_idx: int,
    out_dir: str,
) :
    if (distances.size == 0) : return

    vals = distances[:, rank_idx]
    vals = vals[np.isfinite(vals)]

    if (len(vals) == 0) : return

    plt.figure()
    plt.hist(vals, bins=50)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title(f"{label_name} / {db_type.upper()} DB / rank {rank_idx + 1}")
    plt.tight_layout()

    fname = f"{label_name}_{db_type}_rank{rank_idx + 1}.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved histogram -> {save_path}")


def main() :
    
    result_num = 8
    k = 3
    
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")

    config_path = os.path.join(result_dir, "config.json")
    if (not os.path.exists(config_path)) :
        raise RuntimeError(f"config.json not found in {result_dir}")

    with open(config_path, "r", encoding = "utf-8") as f :
        config = json.load(f)

    model_name = config.get("model_name")

    bad_vec_path = os.path.join(result_dir, "bad_db_vectors.npy")
    good_vec_path = os.path.join(result_dir, "good_db_vectors.npy")

    bad_vectors = np.load(bad_vec_path) if os.path.exists(bad_vec_path) else None
    good_vectors = np.load(good_vec_path) if os.path.exists(good_vec_path) else None

    print(f"[INFO] BAD  DB vectors: {0 if bad_vectors is None else len(bad_vectors)}")
    print(f"[INFO] GOOD DB vectors: {0 if good_vectors is None else len(good_vectors)}")

    test_path = os.path.join("dataset", "ko", "test.csv")
    print(f"[INFO] Loading test dataset :: {test_path}")
    X_test, y_test = load_dataset(test_path)
    y_test = np.array(y_test, dtype=int)
    print(f"[INFO] Test size :: {len(X_test)}")

    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name=model_name, device=None)
    print(f"[INFO] Encoding test texts :: {len(X_test)}")
    test_vectors = embedder.encode(list(X_test))

    print("[INFO] Computing top-k distances to GOOD / BAD DB :: k = {k}")
    topk_good = compute_topk_distances(test_vectors, good_vectors, k=k)
    topk_bad = compute_topk_distances(test_vectors, bad_vectors, k=k)

    analytics_dir = os.path.join(result_dir, "analytics_distances")
    os.makedirs(analytics_dir, exist_ok = True)

    np.save(os.path.join(analytics_dir, "topk_good.npy"), topk_good)
    np.save(os.path.join(analytics_dir, "topk_bad.npy"), topk_bad)
    np.save(os.path.join(analytics_dir, "y_test.npy"), y_test)
    print(f"[INFO] Saved raw distance arrays :: {analytics_dir}")

    for label_value, label_name in [(1, "label1"), (0, "label0")] :
        mask = (y_test == label_value)

        bad_dists_label = topk_bad[mask]  
        good_dists_label = topk_good[mask]

        for rank_idx in range(k) :
            plot_histogram(
                bad_dists_label,
                label_name = label_name,
                db_type = "bad",
                rank_idx = rank_idx,
                out_dir = analytics_dir,
            )
            
            plot_histogram(
                good_dists_label,
                label_name = label_name,
                db_type = "good",
                rank_idx = rank_idx,
                out_dir = analytics_dir,
            )

    print("[INFO] Distance distribution analysis finished.")


if (__name__ == "__main__") :
    main()
