import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def evaluate (y_true : np.ndarray, y_pred : np.ndarray, path : str) -> None :
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    os.makedirs(os.path.dirname(path), exist_ok = True)
    
    with open(path, "w") as f :
        f.write(f"Accuracy :: {accuracy:.4f}\n")
        f.write(f"Precision :: {precision:.4f}\n")
        f.write(f"Recall :: {recall:.4f}\n")
        f.write(f"F1 Score :: {f1:.4f}\n")
        f.write(report)        
        f.write("\n\n")

        f.write("Confusion Matrix\n")
        f.write("                       Real Label\n")
        f.write("                 Positive   Negative\n")
        f.write(f"Pred Positive   TP={tp:<6} FP={fp:<6}\n")
        f.write(f"Pred Negative   FN={fn:<6} TN={tn:<6}\n\n")
        
        
def write_time (time : float, path : str, message : str = "") -> None :
    if not os.path.exists(path) :
        with open(path, "w") as f :
            f.write(f"{message} :: {time:.2f} seconds\n")
    else :
        with open(path, "a") as f :
            f.write(f"{message} :: {time:.2f} seconds\n")

def write_config (config : dict, path : str) -> None :
    if not os.path.exists(path) :
        with open(path, "w", encoding="utf-8") as f :
            f.write(json.dumps(config, ensure_ascii=False, indent=2))
    else :
        with open(path, "a", encoding="utf-8") as f :
            f.write(json.dumps(config, ensure_ascii=False, indent=2))