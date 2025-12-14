import os
import time
from dotenv import load_dotenv

from typing import List

import numpy as np
from openai import OpenAI

from src.core.data import load_dataset
from src.core.metrics import evaluate, write_time

TEST_PATH = os.path.join("dataset", "test.csv")
RESULTS_DIR = os.path.join("results", "comparison_gpt4_1_nano")
PROMPT_PATH = os.path.join("src", "experiments", "comparison_gpt4_1_nano", "prompt.txt")

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4.1-nano"


def parse_response (response : str) -> int :
    text = (response or "").strip()

    if (text == "1") : return 1  # bad comment
    if (text == "0") : return 0  # good comment

    digits = [ch for ch in text if ch in ("0", "1")]
    if digits : return int(digits[0])

    return None
    
    
def predict (texts : List[str]) -> np.ndarray :
    
    client = OpenAI(api_key = api_key)
    prompt = open(PROMPT_PATH, "r").read()
    
    times = []
    predictions = []
    
    for i, text in enumerate(texts) :
        
        if (i % 100 == 0) : print(f"[INFO] Processing :: {i}")
        
        start_time = time.perf_counter() # time start
        
        response = client.chat.completions.create(
            model = model,
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"댓글 : {text}"},
            ]
        )
        
        end_time = time.perf_counter() # time end
        elapsed = end_time - start_time
        times.append(elapsed)

        predictions.append(parse_response(response.choices[0].message.content))
    
    times_path = os.path.join(RESULTS_DIR, f"inference_times.npy")
    np.save(times_path, np.array(times, dtype = float))
    print(f"[INFO] Saved inference times :: {times_path}")
    
    predictions_path = os.path.join(RESULTS_DIR, f"predictions.npy")
    np.save(predictions_path, np.array(predictions, dtype = int))
    print(f"[INFO] Saved predictions :: {predictions_path}")
    
    write_time(np.mean(times), os.path.join(RESULTS_DIR, "time.txt"), "Average Inference Time")
    write_time(np.sum(times), os.path.join(RESULTS_DIR, "time.txt"), "Total Inference Time")
    
    return np.array(predictions, dtype = int)


def main() :
    
    os.makedirs(RESULTS_DIR, exist_ok = True)
    
    # 1. test set 로드
    print(f"[INFO] Loading test dataset")
    X_test, y_test = load_dataset(TEST_PATH)
    print(f"[INFO] Test size :: {len(X_test)}")

    # 2. GPT-4.1-nano로 분류
    print("[INFO] Classifying test set :: gpt-4.1-nano")
    y_pred = predict(list(X_test))

    if len(y_pred) != len(y_test) :
        raise RuntimeError(f"Prediction count :: {len(y_pred)} does not match the answer count :: {len(y_test)}")

    # 3. 동일 metric으로 평가
    evaluate(y_test, y_pred, os.path.join(RESULTS_DIR, "metrics.txt"))
    

if (__name__ == "__main__") :
    main()