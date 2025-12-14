import os
import time
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.data import load_dataset
from src.core.metrics import evaluate, write_time


dataset = "en"
TEST_PATH = os.path.join("dataset", dataset, "test.csv")
RESULTS_DIR = os.path.join("results", dataset, "comparison_llamaguard_3_1b")

MODEL_ID = "meta-llama/Llama-Guard-3-1B"

def get_device_and_dtype () :
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_llamaguard () :
    device, dtype = get_device_and_dtype()
    print(f"[INFO] Loading Llama Guard 3 ({MODEL_ID}) on device = {device}, dtype = {dtype} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = dtype,
        device_map = device,
    )

    print("[INFO] Llama Guard 3 loaded.")
    return tokenizer, model, device


def moderate_one (
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> str :
    
    chat = [
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat,
        return_tensors = "pt"
    ).to(device)

    output = model.generate(
        input_ids = input_ids,
        max_new_tokens = 64,
        pad_token_id = tokenizer.eos_token_id,
    )

    prompt_len = input_ids.shape[-1]
    generated_ids = output[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens = True)

    # llama guard test
    # print(f"generated_text :: {generated_text}")
    
    return generated_text


def parse_response (generated_text : str) -> int :

    if not generated_text : return 1

    s = generated_text.strip().lower()
    if not s : return 1

    first_token = s.split()[0]
    if "unsafe" in first_token : return 1
    if "safe" in first_token : return 0

    if "unsafe" in s : return 1
    if "safe" in s : return 0

    return 1


def predict (
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> np.ndarray :

    times: List[float] = []
    preds: List[int] = []

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"[INFO] Processing :: {i}")

        start_time = time.perf_counter()

        generated = moderate_one(text, tokenizer, model, device)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

        label = parse_response(generated)
        preds.append(label)
        
        # llama guard test
        # print(f"label :: {label}")

    times_arr = np.array(times, dtype=float)
    preds_arr = np.array(preds, dtype=int)

    os.makedirs(RESULTS_DIR, exist_ok = True)

    times_path = os.path.join(RESULTS_DIR, "inference_times.npy")
    np.save(times_path, times_arr)
    print(f"[INFO] Saved inference times :: {times_path}")

    preds_path = os.path.join(RESULTS_DIR, "predictions.npy")
    np.save(preds_path, preds_arr)
    print(f"[INFO] Saved predictions :: {preds_path}")

    write_time(float(np.mean(times_arr)), os.path.join(RESULTS_DIR, "time.txt"), "Average Inference Time")
    write_time(float(np.sum(times_arr)),  os.path.join(RESULTS_DIR, "time.txt"), "Total Inference Time")

    return preds_arr


def main () :
    os.makedirs(RESULTS_DIR, exist_ok = True)

    # 1. test set 로드
    print(f"[INFO] Loading test dataset from {TEST_PATH}")
    X_test, y_test = load_dataset(TEST_PATH)
    print(f"[INFO] Test size :: {len(X_test)}")

    # 2. Llama Guard 로드
    tokenizer, model, device = load_llamaguard()

    # 3. 분류
    print("[INFO] Classifying test set :: Llama-Guard-3-1B")
    y_pred = predict(list(X_test), tokenizer, model, device)

    if len(y_pred) != len(y_test):
        raise RuntimeError(
            f"Prediction count :: {len(y_pred)} does not match the answer count :: {len(y_test)}"
        )

    # 4. 동일 metric으로 평가
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    evaluate(y_test, y_pred, metrics_path)


if (__name__ == "__main__") :
    main()
