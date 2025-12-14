import os
import time

import numpy as np
import pandas as pd

from src.core.data import load_dataset, plot_threshold_history, find_numbering
from src.core.embedding import Embedder
from src.experiments.ko.experiment01_bad_db_only.model import AdaptiveBadDBClassifier
from src.core.metrics import evaluate, write_time, write_config

dataset = "ko"
TRAIN_PATH = os.path.join("dataset", dataset, "train.csv")
TEST_PATH = os.path.join("dataset", dataset, "test.csv")
RESULTS_DIR = os.path.join("results", dataset, "experiment01_bad_db_only")


def main () :
    
    numbering = find_numbering(RESULTS_DIR)
    result_path = os.path.join(RESULTS_DIR, numbering)
    os.makedirs(result_path, exist_ok = True)
    
    # 1. 훈련 데이터 로드
    print(f"[INFO] Loading dataset from {TRAIN_PATH} ...")
    X_train, y_train = load_dataset(TRAIN_PATH)
    print(f"[INFO] Train size: {len(X_train)}")
    

    # 2. 임베딩 모델 준비
    print(f"[INFO] Loading embedding model ...")
    model_name = "MongoDB/mdbr-leaf-mt"
    embedder = Embedder(
        model_name = model_name,
        device = None
    )
    print(f"[INFO] Embedding model loaded")

    # 3. AdaptiveBadDBClassifier 학습
    print(f"[INFO] Training AdaptiveBadDBClassifier ...")
    start_time = time.perf_counter()
    clf = AdaptiveBadDBClassifier(
        initial_threshold=0.35,
        min_threshold=0.1,
        max_threshold=1.0,
        threshold_step=0.0001,
        seed_size=10,
        random_state=1,
    )
    print(f"[INFO] AdaptiveBadDBClassifier trained")
    clf.fit(X_train, y_train, embedder=embedder)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    write_time(elapsed, os.path.join(result_path, "time.txt"), "Training")

    # 4. 학습 결과 저장    
    # (1) config 저장
    config = {
        "model_name": model_name,
        "initial_threshold": clf.initial_threshold,
        "min_threshold": clf.min_threshold,
        "max_threshold": clf.max_threshold,
        "threshold_step": clf.threshold_step
    }
    write_config(config, os.path.join(result_path, "config.json"))
    
    # (2) threshold 변화 그래프 출력/저장
    print(f"[INFO] Saving threshold history ...")
    history = clf.get_threshold_history()
    out_path = os.path.join(result_path, "threshold_history.png")
    plot_threshold_history(history, out_path=out_path)
    
    # (3) BAD DB 텍스트 저장
    print(f"[INFO] Saving BAD DB texts ...")
    if clf.bad_texts :
        df_bad = pd.DataFrame({
            "content": clf.bad_texts
        })
        bad_csv_path = os.path.join(result_path, "bad_db.csv")
        df_bad.to_csv(bad_csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] BAD DB texts saved to {bad_csv_path}")

    # (4) BAD DB 벡터 저장 (.npy)
    print(f"[INFO] Saving BAD DB vectors ...")
    if clf.bad_vectors is not None:
        bad_vec_path = os.path.join(result_path, "bad_db_vectors.npy")
        np.save(bad_vec_path, clf.bad_vectors)
        print(f"[INFO] BAD DB vectors saved to {bad_vec_path}")

    
    # 5. 테스트 데이터 로드 및 평가
    print(f"[INFO] Evaluating on test set...")
    print(f"[INFO] Loading dataset from {TEST_PATH} ...")
    X_test, y_test = load_dataset(TEST_PATH)
    print(f"[INFO] Test size: {len(X_test)}")
    
    print(f"[INFO] Predicting on test set ...")
    start_time = time.perf_counter()
    y_pred = clf.predict(X_test, embedder = embedder)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    write_time(elapsed, os.path.join(result_path, "time.txt"), "Prediction")

    # 6. 평가
    print(f"[INFO] Evaluating on test set ...")
    evaluate(y_test, y_pred, os.path.join(result_path, "metrics.txt"))
    print(f"[INFO] Evaluation completed")


if (__name__ == "__main__") :
    main()
