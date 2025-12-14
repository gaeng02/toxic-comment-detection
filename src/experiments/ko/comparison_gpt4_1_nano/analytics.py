import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.data import load_dataset
from src.core.metrics import evaluate, write_time

RESULTS_DIR = os.path.join("results", "comparison_gpt4_1_nano")
TEST_PATH = os.path.join("dataset", "test.csv")


def main():
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 추론 시간 분석 및 time.txt 작성
    times_path = os.path.join(RESULTS_DIR, "gpt-4.1-nano_inference_times.npy")
    time_txt_path = os.path.join(RESULTS_DIR, "time.txt")
    
    if os.path.exists(times_path):
        times = np.load(times_path)
        avg_time = np.mean(times)
        total_time = np.sum(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # time.txt 작성 (다른 실험과 동일한 형식)
        write_time(total_time, time_txt_path, "Prediction")
        
        print("=" * 60)
        print("추론 시간 분석 (Inference Time Analysis)")
        print("=" * 60)
        print(f"총 시간 (Total Time)     : {total_time:.4f} 초")
        print(f"평균 시간 (Average Time) : {avg_time:.4f} 초")
        print(f"최소 시간 (Min Time)     : {min_time:.4f} 초")
        print(f"최대 시간 (Max Time)     : {max_time:.4f} 초")
        print(f"표준편차 (Std Dev)       : {std_time:.4f} 초")
        print(f"총 샘플 수 (Total Samples): {len(times)}")
        print(f"[INFO] time.txt 저장 완료: {time_txt_path}")
        print()
    else:
        print(f"[WARNING] 추론 시간 파일을 찾을 수 없습니다: {times_path}")
        print()
    
    # 2. 예측 결과 분석 및 metrics.txt 작성
    predictions_path = os.path.join(RESULTS_DIR, "gpt-4.1-nano_predictions.npy")
    metrics_txt_path = os.path.join(RESULTS_DIR, "metrics.txt")
    
    if os.path.exists(predictions_path):
        y_pred = np.load(predictions_path)
        
        # 테스트 데이터 로드
        print(f"[INFO] 테스트 데이터 로드 중: {TEST_PATH}")
        _, y_test = load_dataset(TEST_PATH)
        
        # 길이 확인 및 조정
        min_len = min(len(y_pred), len(y_test))
        if len(y_pred) != len(y_test):
            print(f"[WARNING] 예측 개수 ({len(y_pred)})와 정답 개수 ({len(y_test)})가 다릅니다.")
            print(f"[WARNING] 처음 {min_len}개만 사용합니다.")
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
        
        # metrics.txt 작성 (다른 실험과 동일한 형식)
        evaluate(y_test, y_pred, metrics_txt_path)
        print(f"[INFO] metrics.txt 저장 완료: {metrics_txt_path}")
        print()
        
        # 메트릭 계산 (출력용)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print("=" * 60)
        print("성능 메트릭 (Performance Metrics)")
        print("=" * 60)
        print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
        print(f"총 샘플 수 (Total Samples): {len(y_test)}")
        print()
        
        # 클래스별 분포
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print("=" * 60)
        print("클래스 분포 (Class Distribution)")
        print("=" * 60)
        print("정답 (Ground Truth):")
        for label, count in zip(unique_test, counts_test):
            print(f"  클래스 {label}: {count}개 ({count/len(y_test)*100:.2f}%)")
        print("예측 (Predictions):")
        for label, count in zip(unique_pred, counts_pred):
            print(f"  클래스 {label}: {count}개 ({count/len(y_pred)*100:.2f}%)")
        print()
        
    else:
        print(f"[WARNING] 예측 결과 파일을 찾을 수 없습니다: {predictions_path}")
        print()


if (__name__ == "__main__") :
    main()

