import pandas as pd
import matplotlib.pyplot as plt
import os

def load_dataset (path : str) -> pd.DataFrame :

    df = pd.read_csv(path)

    if (len(df.columns) == 1) and ("\t" in df.columns[0]) :
        df = pd.read_csv(path, sep = "\t")

    if ("content" not in df.columns) : raise ValueError("'content' column is required")
    if ("label" not in df.columns) : raise ValueError("'label' column is required")

    df["label"] = pd.to_numeric(df["label"], errors = "coerce")

    before = len(df)
    df = df.dropna(subset = ["label"])
    dropped = before - len(df)
    if (dropped > 0) : print(f"[WARN] {dropped} samples are dropped due to conversion error")

    df["label"] = df["label"].astype(int)

    return df["content"], df["label"]


def find_numbering (directory : str) -> str :
    
    if not os.path.isdir(directory) :
        os.makedirs(directory, exist_ok = True)
        return "01"

    max_num = 0
    
    for name in os.listdir(directory) :
        if name.isdigit() :
            n = int(name)
            if n > max_num : max_num = n

    return f"{max_num + 1:02d}"


def plot_threshold_history(history, out_path: str = None) -> None :
    plt.figure()
    plt.plot(history)
    plt.xlabel("Training step")
    plt.ylabel("Threshold (tau)")
    plt.title("Adaptive threshold over training")
    plt.grid(True)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[INFO] Threshold history plot saved to {out_path}")
    else:
        plt.show()