import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join("dataset", "dataset.csv")
TRAIN_PATH = os.path.join("dataset", "train.csv")
TEST_PATH = os.path.join("dataset", "test.csv")


def load_raw_dataset (path : str = DATASET_PATH) -> pd.DataFrame :

    df = pd.read_csv(path)

    if len(df.columns) == 1 and "\t" in df.columns[0] : df = pd.read_csv(path, sep="\t")

    if "content" not in df.columns : raise ValueError("'content' column is required")

    if "label" not in df.columns : raise ValueError("'label' column is required")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["label"])
    dropped = before - len(df)
    if dropped > 0 :
        print(f"[WARN] {dropped} samples are dropped due to conversion error")

    df["label"] = 1 - df["label"].astype(int) # label 반전

    return df


def main () :
    
    # 1. 원본 데이터 로드
    print(f"[INFO] Loading raw dataset from {DATASET_PATH} ...")
    df = load_raw_dataset(DATASET_PATH)
    print(f"[INFO] Total valid samples: {len(df)}")
    print(df["label"].value_counts())

    # 2. train / test 분리
    train_df, test_df = train_test_split(
        df,
        test_size = 0.2,
        random_state = 1,
        stratify=df["label"],
    )

    # 3. 저장
    os.makedirs("dataset", exist_ok = True)

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] Train saved to {TRAIN_PATH} (n={len(train_df)})")
    print(f"[INFO] Test  saved to {TEST_PATH} (n={len(test_df)})")


if (__name__ == "__main__") :
    main()