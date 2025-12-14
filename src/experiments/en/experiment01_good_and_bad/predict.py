import os
import time

import numpy as np
import pandas as pd

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.experiments.en.experiment01_good_and_bad.model import GoodBadKNNClassifier
from src.core.metrics import evaluate, write_time, write_config

dataset = "en"
TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")
TEST_PATH = os.path.join("dataset", dataset, "test.csv")
RESULTS_DIR = os.path.join("results", dataset, "experiment01_good_and_bad")


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
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = Embedder(
        model_name = model_name,
        device = None
    )
    print(f"[INFO] Embedding model loaded")

    # 3. AdaptiveBadDBClassifier 학습
    print(f"[INFO] Training GoodBadKNNClassifier ...")
    start_time = time.perf_counter()
    clf = GoodBadKNNClassifier(
        seed_size = 100,
        k = 49,
        random_state = 1,
    )
    clf.fit(X_train, y_train, embedder = embedder)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    write_time(elapsed, os.path.join(result_path, "time.txt"), "Training")

    # 4. 학습 결과 저장    
    # (1) config 저장
    config = {
        "model_name": model_name,
        "dataset_language" : dataset, 
        "seed_size": clf.seed_size,
        "k": clf.k,
    }
    write_config(config, os.path.join(result_path, "config.json"))
    
    # (2) GOOD / BAD DB 텍스트 저장
    print(f"[INFO] Saving BAD DB texts ...")
    if clf.bad_texts :
        df_bad = pd.DataFrame({
            "content": clf.bad_texts
        })
        bad_csv_path = os.path.join(result_path, "bad_db.csv")
        df_bad.to_csv(bad_csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] BAD DB texts saved to {bad_csv_path}")
    
    print(f"[INFO] Saving GOOD DB texts ...")
    if clf.good_texts :
        df_good = pd.DataFrame({
            "content": clf.good_texts
        })
        good_csv_path = os.path.join(result_path, "good_db.csv")
        df_good.to_csv(good_csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] GOOD DB texts saved to {good_csv_path}")

    # (3) GOOD / BAD DB 벡터 저장 (.npy)
    print(f"[INFO] Saving BAD DB vectors ...")
    if clf.bad_vectors is not None:
        bad_vec_path = os.path.join(result_path, "bad_db_vectors.npy")
        np.save(bad_vec_path, clf.bad_vectors)
        print(f"[INFO] BAD DB vectors saved to {bad_vec_path}")

    print(f"[INFO] Saving GOOD DB vectors ...")
    if clf.good_vectors is not None:
        good_vec_path = os.path.join(result_path, "good_db_vectors.npy")
        np.save(good_vec_path, clf.good_vectors)
        print(f"[INFO] GOOD DB vectors saved to {good_vec_path}")
    

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
