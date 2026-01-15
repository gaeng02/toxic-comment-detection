import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.core.metrics import evaluate, write_time, write_config
from src.experiments.en.experiment05_clustering.model import (
    build_knn_features,
    DistanceNN,
    DistanceNNConfig,
)

# config 
k = 20
method = "02_dbscan/03"

dataset = "en"

TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")
TEST_PATH  = os.path.join("dataset", dataset, "test.csv")
RESULTS_DIR = os.path.join("results", dataset, "experiment05_clustering", method)

result_path = os.path.join(RESULTS_DIR, f"{k:03d}")
GOOD_DB_PATH = os.path.join(RESULTS_DIR, "good_db_vectors.npy")
BAD_DB_PATH  = os.path.join(RESULTS_DIR, "bad_db_vectors.npy")


def get_device() -> str :
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() :

    os.makedirs(result_path, exist_ok = True)
    print(f"[INFO] Result path :: {result_path}")

    # 1. 데이터 로드
    print(f"[INFO] Loading train dataset from {TRAIN_PATH} ...")
    X_train, y_train = load_dataset(TRAIN_PATH)
    print(f"[INFO] Train size :: {len(X_train)}")

    print(f"[INFO] Loading test dataset from {TEST_PATH} ...")
    X_test, y_test = load_dataset(TEST_PATH)
    print(f"[INFO] Test size :: {len(X_test)}")

    y_train = np.asarray(y_train, dtype=int)
    y_test  = np.asarray(y_test,  dtype=int)

    # 2. DB 벡터 로드
    if not os.path.exists(GOOD_DB_PATH):
        raise RuntimeError(f"GOOD DB vectors not found :: {GOOD_DB_PATH}")
    if not os.path.exists(BAD_DB_PATH):
        raise RuntimeError(f"BAD DB vectors not found :: {BAD_DB_PATH}")

    good_db_vectors = np.load(GOOD_DB_PATH)
    bad_db_vectors  = np.load(BAD_DB_PATH)

    print(f"[INFO] GOOD DB size :: {len(good_db_vectors)}")
    print(f"[INFO] BAD  DB size :: {len(bad_db_vectors)}")

    # 3. 임베딩 모델 준비
    print("[INFO] Loading embedding model ...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = Embedder(
        model_name = model_name,
        device = None,
    )
    print("[INFO] Embedding model loaded.")

    # 4. train 임베딩
    print("[INFO] Encoding train texts ...")
    t0 = time.perf_counter()
    train_vectors = embedder.encode(list(X_train))   # (N_train, D)
    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "Train embedding")

    # 5. k-NN 기반 feature 생성
    k_neighbors = k
    print(f"[INFO] Building k-NN features with k = {k_neighbors} ...")

    t0 = time.perf_counter()
    train_feats = build_knn_features(
        query_vectors   = train_vectors,
        good_db_vectors = good_db_vectors,
        bad_db_vectors  = bad_db_vectors,
        k               = k_neighbors,
    )  # (N_train, 2*k)
    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "Build train features")

    
    # 6. test 임베딩
    print("[INFO] Encoding test texts ...")
    t0 = time.perf_counter()
    test_vectors = embedder.encode(list(X_test))     # (N_test, D)
    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "Test embedding")

    t0 = time.perf_counter()
    test_feats = build_knn_features(
        query_vectors   = test_vectors,
        good_db_vectors = good_db_vectors,
        bad_db_vectors  = bad_db_vectors,
        k               = k_neighbors,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "Build test features")


    def nan_to_col_mean(x: np.ndarray) -> np.ndarray :
        x = x.copy()
        col_mean = np.nanmean(x, axis=0)

        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])
        return x

    train_feats = nan_to_col_mean(train_feats)
    test_feats  = nan_to_col_mean(test_feats)

    # 7. DistanceNN 학습 세팅
    input_dim = train_feats.shape[1]
    config = DistanceNNConfig(
        input_dim   = input_dim,
        hidden_dim1 = 64,
        hidden_dim2 = 32,
        dropout     = 0.1,
    )

    # config 저장
    config_dict = {
        "dataset_language": dataset,
        "embedder_model": model_name,
        "k": k,
        "method": method,
        "input_dim": config.input_dim,
        "hidden_dim1": config.hidden_dim1,
        "hidden_dim2": config.hidden_dim2,
        "dropout": config.dropout,
    }
    write_config(config_dict, os.path.join(result_path, "config.json"))

    device = get_device()
    print(f"[INFO] Training DistanceNN on device :: {device}")

    # train/valid split
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_feats,
        y_train,
        test_size = 0.2,
        random_state = 1,
        stratify = y_train,
    )

    X_tr_tensor  = torch.from_numpy(X_tr).float()
    y_tr_tensor  = torch.from_numpy(y_tr).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    train_ds = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_ds, batch_size = 256, shuffle = True)

    model = DistanceNN(config).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # 8. 학습 루프
    num_epochs = 10
    t0 = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)              # (B,)
            loss = criterion(logits, yb)    # yb: (B,)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(xb)

        epoch_loss /= len(X_tr)

        # validation loss / accuracy
        model.eval()
        with torch.no_grad():
            xb = X_val_tensor.to(device)
            yb = y_val_tensor.to(device)
            logits = model(xb)
            val_loss = criterion(logits, yb).item()

            probs = torch.sigmoid(logits)
            y_hat = (probs >= 0.5).long().cpu().numpy()
            val_acc = (y_hat == y_val).mean()

        print(f"[INFO] Epoch {epoch:02d} | train_loss = {epoch_loss:.4f} | "
              f"val_loss = {val_loss:.4f} | val_acc = {val_acc:.4f}")

    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "NN training")

    # 8. test 평가
    print("[INFO] Evaluating on test set with DistanceNN ...")

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(test_feats).float().to(device)
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)

    t1 = time.perf_counter()
    write_time(t1 - t0, os.path.join(result_path, "time.txt"), "NN test")

    metrics_path = os.path.join(result_path, "metrics.txt")
    evaluate(y_test, y_pred, metrics_path)
    print(f"[INFO] Evaluation completed. Metrics saved to {metrics_path}")

    # 9. 모델 저장 (선택)
    weights_path = os.path.join(result_path, "distance_nn.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"[INFO] Saved DistanceNN weights to {weights_path}")


if (__name__ == "__main__") :
    main()
