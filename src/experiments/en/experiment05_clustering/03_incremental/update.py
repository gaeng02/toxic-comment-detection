"""
Incremental DB 업데이트 스크립트

기존에 구축된 centroid DB에 새로운 데이터를 추가하는 스크립트
"""
import os
import time

import numpy as np

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.core.metrics import write_time, write_config
from .builder import Centroid, find_nearest_centroid


# ===== config =====
dataset = "en"

# 기존 DB 경로
RESULTS_DIR = os.path.join("results", dataset, "experiment05_clustering", "03_incremental")
EXISTING_RESULT_PATH = os.path.join(RESULTS_DIR, "01")  # 기존 결과 디렉토리

# 새 데이터 경로
NEW_DATA_PATH = os.path.join("dataset", dataset, "train_all.csv")  # 예시

# Incremental clustering hyper-parameters (기존과 동일하게)
eps_good = 1.0
eps_bad = 1.0
overlap_threshold = 0.5
min_members_for_split = 10


def load_existing_centroids(result_path: str) -> tuple[list[Centroid], int]:
    """
    기존에 저장된 centroid 정보를 로드하여 Centroid 객체 리스트로 변환
    """
    centroid_vectors_path = os.path.join(result_path, "centroid_vectors.npy")
    centroid_labels_path = os.path.join(result_path, "centroid_labels.npy")
    centroid_radii_path = os.path.join(result_path, "centroid_radii.npy")
    centroid_counts_path = os.path.join(result_path, "centroid_counts.npy")
    centroid_ids_path = os.path.join(result_path, "centroid_ids.npy")
    
    if not os.path.exists(centroid_vectors_path):
        raise FileNotFoundError(f"Centroid vectors not found: {centroid_vectors_path}")
    
    centroid_vectors = np.load(centroid_vectors_path)
    centroid_labels = np.load(centroid_labels_path)
    centroid_radii = np.load(centroid_radii_path)
    centroid_counts = np.load(centroid_counts_path)
    centroid_ids = np.load(centroid_ids_path)
    
    centroids = []
    max_id = -1
    
    for i in range(len(centroid_vectors)):
        centroid = Centroid(
            centroid_id=int(centroid_ids[i]),
            label=int(centroid_labels[i]),
            position=centroid_vectors[i].copy(),
        )
        # 멤버 정보는 복원할 수 없으므로, position만 사용
        # radius는 저장된 값 사용
        centroid.radius = float(centroid_radii[i])
        # 멤버 수 정보는 counts에 저장되어 있지만, 실제 멤버 벡터는 없음
        # 업데이트 시에는 position만 사용하여 점진적으로 업데이트
        centroids.append(centroid)
        
        if centroid.centroid_id > max_id:
            max_id = centroid.centroid_id
    
    print(f"[INFO] Loaded {len(centroids)} existing centroids")
    return centroids, max_id + 1


def update_centroids_incremental(
    centroids: list[Centroid],
    new_vectors: np.ndarray,
    new_labels: np.ndarray,
    target_label: int,
    eps: float,
    next_centroid_id: int,
    overlap_threshold: float = 0.5,
) -> tuple[list[Centroid], int]:
    """
    기존 centroids에 새로운 벡터들을 incremental하게 추가
    
    Returns:
        업데이트된 centroids 리스트와 다음 centroid ID
    """
    mask = (new_labels == target_label)
    class_vectors = new_vectors[mask]
    
    if class_vectors.shape[0] == 0:
        return centroids, next_centroid_id
    
    # 같은 레이블의 centroids만 필터링
    target_centroids = [c for c in centroids if c.label == target_label]
    
    print(f"[INFO] Updating centroids for label {target_label} with {class_vectors.shape[0]} new vectors")
    print(f"[INFO] Existing centroids: {len(target_centroids)}")
    
    for i, vector in enumerate(class_vectors):
        # 가장 가까운 centroid 찾기
        nearest = find_nearest_centroid(vector, target_centroids, eps)
        
        if nearest is not None:
            # 기존 centroid에 추가
            nearest.add_member(vector)
            
            # 분할 조건 확인
            if nearest.should_split(target_centroids, overlap_threshold):
                # 분할 로직 (builder.py와 동일)
                members_array = np.stack(nearest.members, axis=0)
                if len(members_array) >= 2:
                    distances = np.linalg.norm(
                        members_array - nearest.position, axis=1
                    )
                    far_idx = int(np.argmax(distances))
                    far_point = members_array[far_idx]
                    
                    new_centroid = Centroid(
                        centroid_id=next_centroid_id,
                        label=target_label,
                        position=far_point.copy(),
                    )
                    new_centroid.members = [far_point.copy()]
                    new_centroid._recalculate_radius()
                    
                    nearest.members = [
                        m for j, m in enumerate(nearest.members) if j != far_idx
                    ]
                    nearest._recalculate_position()
                    nearest._recalculate_radius()
                    
                    target_centroids.append(new_centroid)
                    centroids.append(new_centroid)
                    next_centroid_id += 1
        else:
            # 새 centroid 생성
            new_centroid = Centroid(
                centroid_id=next_centroid_id,
                label=target_label,
                position=vector.copy(),
            )
            target_centroids.append(new_centroid)
            centroids.append(new_centroid)
            next_centroid_id += 1
        
        if (i + 1) % 1000 == 0:
            print(f"[INFO] Processed {i + 1}/{class_vectors.shape[0]} new samples, "
                  f"centroids = {len(target_centroids)}")
    
    print(f"[INFO] Updated centroids for label {target_label}: {len(target_centroids)}")
    return centroids, next_centroid_id


def main():
    # 새 결과 디렉토리 생성
    numbering = find_numbering(RESULTS_DIR)
    result_path = os.path.join(RESULTS_DIR, numbering)
    os.makedirs(result_path, exist_ok=True)
    print(f"[INFO] Result path :: {result_path}")
    
    # 저장 경로
    CENTROID_VECTORS_PATH = os.path.join(result_path, "centroid_vectors.npy")
    CENTROID_LABELS_PATH = os.path.join(result_path, "centroid_labels.npy")
    CENTROID_RADII_PATH = os.path.join(result_path, "centroid_radii.npy")
    CENTROID_COUNTS_PATH = os.path.join(result_path, "centroid_counts.npy")
    CENTROID_IDS_PATH = os.path.join(result_path, "centroid_ids.npy")
    
    GOOD_DB_PATH = os.path.join(result_path, "good_db_vectors.npy")
    BAD_DB_PATH = os.path.join(result_path, "bad_db_vectors.npy")
    
    DB_VECTORS_PATH = os.path.join(result_path, "db_vectors.npy")
    DB_LABELS_PATH = os.path.join(result_path, "db_labels.npy")
    DB_COUNTS_PATH = os.path.join(result_path, "db_counts.npy")
    
    CONFIG_PATH = os.path.join(result_path, "config.json")
    TIME_PATH = os.path.join(result_path, "time.txt")
    
    # 1. 기존 centroid 로드
    print(f"[INFO] Loading existing centroids from {EXISTING_RESULT_PATH} ...")
    existing_centroids, next_centroid_id = load_existing_centroids(EXISTING_RESULT_PATH)
    print(f"[INFO] Next centroid ID will start from {next_centroid_id}")
    
    # 2. 새 데이터 로드
    print(f"[INFO] Loading new data from {NEW_DATA_PATH} ...")
    X_new, y_new = load_dataset(NEW_DATA_PATH)
    print(f"[INFO] New data size :: {len(X_new)}")
    
    y_new = np.asarray(y_new, dtype=int)
    
    # 3. 임베딩 모델 로드
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(
        model_name=model_name,
        device=None,
    )
    print("[INFO] Embedding model loaded.")
    
    # 4. 새 데이터 임베딩
    print("[INFO] Encoding new texts ...")
    t0 = time.perf_counter()
    new_vectors = embedder.encode(list(X_new))
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "New data embedding")
    print(f"[INFO] New data embedding done :: shape = {new_vectors.shape}")
    
    # 5. Incremental 업데이트
    print("[INFO] Updating centroids for GOOD (label = 0) ...")
    t0 = time.perf_counter()
    updated_centroids, next_centroid_id = update_centroids_incremental(
        centroids=existing_centroids.copy(),
        new_vectors=new_vectors,
        new_labels=y_new,
        target_label=0,
        eps=eps_good,
        next_centroid_id=next_centroid_id,
        overlap_threshold=overlap_threshold,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Incremental update (good)")
    
    print("[INFO] Updating centroids for BAD (label = 1) ...")
    t0 = time.perf_counter()
    updated_centroids, next_centroid_id = update_centroids_incremental(
        centroids=updated_centroids,
        new_vectors=new_vectors,
        new_labels=y_new,
        target_label=1,
        eps=eps_bad,
        next_centroid_id=next_centroid_id,
        overlap_threshold=overlap_threshold,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Incremental update (bad)")
    
    # 6. 업데이트된 centroid 정보 저장
    good_centroids = [c for c in updated_centroids if c.label == 0]
    bad_centroids = [c for c in updated_centroids if c.label == 1]
    
    centroid_vectors = np.stack([c.position for c in updated_centroids], axis=0).astype(np.float32)
    centroid_labels = np.asarray([c.label for c in updated_centroids], dtype=int)
    centroid_radii = np.asarray([c.radius for c in updated_centroids], dtype=np.float32)
    centroid_counts = np.asarray([c.get_member_count() for c in updated_centroids], dtype=int)
    centroid_ids = np.asarray([c.centroid_id for c in updated_centroids], dtype=int)
    
    print(f"[INFO] Final centroids :: {len(updated_centroids)} "
          f"(good = {len(good_centroids)}, bad = {len(bad_centroids)})")
    
    np.save(CENTROID_VECTORS_PATH, centroid_vectors)
    np.save(CENTROID_LABELS_PATH, centroid_labels)
    np.save(CENTROID_RADII_PATH, centroid_radii)
    np.save(CENTROID_COUNTS_PATH, centroid_counts)
    np.save(CENTROID_IDS_PATH, centroid_ids)
    
    good_db_vectors = centroid_vectors[centroid_labels == 0]
    bad_db_vectors = centroid_vectors[centroid_labels == 1]
    
    np.save(GOOD_DB_PATH, good_db_vectors)
    np.save(BAD_DB_PATH, bad_db_vectors)
    
    db_vectors = centroid_vectors
    db_labels = centroid_labels
    db_counts = centroid_counts
    
    np.save(DB_VECTORS_PATH, db_vectors)
    np.save(DB_LABELS_PATH, db_labels)
    np.save(DB_COUNTS_PATH, db_counts)
    
    # 7. config 저장
    config = {
        "dataset_language": dataset,
        "method": "03_incremental",
        "embedder_model": model_name,
        "eps_good": eps_good,
        "eps_bad": eps_bad,
        "overlap_threshold": overlap_threshold,
        "min_members_for_split": min_members_for_split,
        "centroid_count": int(len(updated_centroids)),
        "centroid_count_good": int(len(good_centroids)),
        "centroid_count_bad": int(len(bad_centroids)),
        "db_dim": int(centroid_vectors.shape[1]),
        "updated_from": EXISTING_RESULT_PATH,
    }
    write_config(config, CONFIG_PATH)
    
    print("[INFO] 03_incremental DB update finished.")


if __name__ == "__main__":
    main()
