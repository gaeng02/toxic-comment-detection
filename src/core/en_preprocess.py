import os

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH  = os.path.join("dataset", "en", "original.csv")
TRAIN_PATH  = os.path.join("dataset", "en", "train.csv")
TEST_PATH   = os.path.join("dataset", "en", "test.csv")

def main () :
    
    print(f"[INFO] Loading dataset from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    if "comment_text" not in df.columns or "toxic" not in df.columns :
        raise ValueError("'comment_text' and 'toxic' columns are required")

    df_out = pd.DataFrame({
        "content": df["comment_text"].astype(str),
        "label": df["toxic"].astype(int),
    })

    print(f"[INFO] Total samples :: {len(df_out)}")
    print(df_out["label"].value_counts())

    train_df, test_df = train_test_split(
        df_out,
        test_size = 0.2,
        random_state = 1,
        stratify = df_out["label"],
    )

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] Train saved to {TRAIN_PATH} (n = {len(train_df)})")
    print(f"[INFO] Test  saved to {TEST_PATH} (n = {len(test_df)})")


if (__name__ == "__main__") :
    main()
