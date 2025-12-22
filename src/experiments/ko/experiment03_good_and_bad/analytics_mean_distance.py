import os
import json

import numpy as np
import matplotlib.pyplot as plt

from src.core.data import load_dataset
from src.core.embedding import Embedder

BASE_RESULTS_DIR = os.path.join("results", "ko", "experiment03_good_and_bad")


def compute_mean_distances (
    test_vectors : np.ndarray,
    db_vectors   : np.ndarray | None,
) -> np.ndarray :

    n_test = test_vectors.shape[0]

    if (db_vectors is None) or (len(db_vectors) == 0) :
        return np.full((n_test,), np.nan, dtype = float)

    # (N_test, N_db)
    dists = np.linalg.norm(
        test_vectors[:, None, :] - db_vectors[None, :, :],
        axis = 2,
    )
    # test 하나당 평균 거리
    mean_dists = dists.mean(axis = 1)
    return mean_dists


def plot_hist (
    values   : np.ndarray,
    title    : str,
    filename : str,
    out_dir  : str,
) -> None :
    
    vals = values[np.isfinite(values)]
    if (len(vals) == 0) :
        print(f"[WARN] No finite values for {title}, skip.")
        return

    plt.figure()
    plt.hist(vals, bins = 50)
    plt.xlabel("Average distance")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved histogram -> {save_path}")


def main () :
    
    result_num = 8
    
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{result_num:02d}")
    print(f"[INFO] Result directory :: {result_dir}")

    config_path = os.path.join(result_dir, "config.json")
    if (not os.path.exists(config_path)) :
        raise RuntimeError(f"config.json not found in {result_dir}")

    with open(config_path, "r", encoding = "utf-8") as f :
        config = json.load(f)

    model_name        = config.get("model_name")

    print(f"[INFO] model_name        :: {model_name}")

    # 2. GOOD / BAD DB 벡터 로드
    bad_vec_path  = os.path.join(result_dir, "bad_db_vectors.npy")
    good_vec_path = os.path.join(result_dir, "good_db_vectors.npy")

    bad_vectors  = np.load(bad_vec_path)  if os.path.exists(bad_vec_path)  else None
    good_vectors = np.load(good_vec_path) if os.path.exists(good_vec_path) else None

    print(f"[INFO] BAD  DB vectors :: {0 if bad_vectors  is None else len(bad_vectors)}")
    print(f"[INFO] GOOD DB vectors :: {0 if good_vectors is None else len(good_vectors)}")

    test_path = os.path.join("dataset", "ko", "test.csv")
    print(f"[INFO] Loading test dataset :: {test_path}")
    X_test, y_test = load_dataset(test_path)
    y_test = np.array(y_test, dtype = int)
    print(f"[INFO] Test size :: {len(X_test)}")

    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(model_name = model_name, device = None)
    print(f"[INFO] Encoding test texts :: {len(X_test)}")
    test_vectors = embedder.encode(list(X_test))

    print("[INFO] Computing mean distances ...")
    mean_good = compute_mean_distances(test_vectors, good_vectors)
    mean_bad  = compute_mean_distances(test_vectors, bad_vectors)

    if (good_vectors is not None) and (bad_vectors is not None) :
        all_db_vectors = np.concatenate([good_vectors, bad_vectors], axis = 0)
    elif good_vectors is not None :
        all_db_vectors = good_vectors
    elif bad_vectors is not None :
        all_db_vectors = bad_vectors
    else :
        all_db_vectors = None

    mean_all = compute_mean_distances(test_vectors, all_db_vectors)

    analytics_dir = os.path.join(result_dir, "analytics_mean_distance")
    os.makedirs(analytics_dir, exist_ok = True)

    plot_hist(
        mean_all,
        title    = "All test / ALL DB average distance",
        filename = "hist_all_test_all_db.png",
        out_dir  = analytics_dir,
    )

    for label_value, label_name in [(1, "label1"), (0, "label0")] :
        mask = (y_test == label_value)
        
        mean_good_label = mean_good[mask]
        mean_bad_label  = mean_bad[mask]

        plot_hist(
            mean_good_label,
            title    = f"{label_name} / GOOD DB average distance",
            filename = f"hist_{label_name}_good_mean.png",
            out_dir  = analytics_dir,
        )
        
        plot_hist(
            mean_bad_label,
            title    = f"{label_name} / BAD DB average distance",
            filename = f"hist_{label_name}_bad_mean.png",
            out_dir  = analytics_dir,
        )

    print("[INFO] Mean distance histogram analysis finished.")


if (__name__ == "__main__") :
    main()
