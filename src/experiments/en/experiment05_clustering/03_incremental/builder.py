import os
import time

import numpy as np
from sklearn.cluster import KMeans

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.core.metrics import write_time, write_config


# ===== config =====
dataset = "en"

# Incremental clustering hyper-parameters
eps_good = 1.0
eps_bad  = 1.0

# 겹침 영역 비율 임계값 (0.0 ~ 1.0)
# 두 centroid의 겹침 영역 비율이 이 값 이상이면 분할
overlap_ratio_threshold = 0.3

method = "03_incremental"
TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")

RESULTS_DIR = os.path.join("results", dataset, "experiment05_clustering", method)


class Centroid :

    def __init__(self, centroid_id: int, label: int, position: np.ndarray):
        self.centroid_id = centroid_id
        self.label = label
        self.position = position.copy()
        self.members = [position.copy()]
        self.radius = 0.0
    
    def add_member(self, vector: np.ndarray):
        self.members.append(vector.copy())
        self._recalculate_position()
        self._recalculate_radius()
    
    def _recalculate_position(self):
        if len(self.members) == 0:
            return
        self.position = np.mean(np.stack(self.members, axis=0), axis=0)
    
    def _recalculate_radius(self):
        if len(self.members) == 0:
            self.radius = 0.0
            return
        
        members_array = np.stack(self.members, axis=0)
        distances = np.linalg.norm(members_array - self.position, axis=1)
        self.radius = float(np.max(distances))
    
    def get_member_count(self) -> int:
        return len(self.members)
    
    def find_overlapping_centroids(self, other_centroids: list['Centroid'], overlap_ratio_threshold: float) -> list['Centroid']:
        """
        겹치는 다른 centroid들을 찾아서 반환
        같은 label이든 다른 label이든 상관없이 겹치는 모든 centroid 반환
        
        겹침 영역 비율 = (r1 + r2 - distance) / min(r1, r2)
        두 구(sphere)가 겹칠 때: distance < (r1 + r2)
        """
        overlapping = []
        
        if len(self.members) < 2:
            return overlapping
        
        for other in other_centroids:
            # 자기 자신만 스킵 (같은 label이든 다른 label이든 모두 체크)
            if other.centroid_id == self.centroid_id:
                continue
            
            distance = np.linalg.norm(self.position - other.position)
            r1, r2 = self.radius, other.radius
            
            # 겹치는지 확인
            if distance >= (r1 + r2):
                continue
            
            # 겹침 영역 비율 계산
            # 겹침 깊이 = (r1 + r2) - distance
            overlap_depth = (r1 + r2) - distance
            min_radius = min(r1, r2)
            
            if min_radius > 0:
                overlap_ratio = overlap_depth / min_radius
            else:
                overlap_ratio = 0.0
            
            if overlap_ratio >= overlap_ratio_threshold:
                overlapping.append(other)
        
        return overlapping


def find_nearest_centroid(
    vector: np.ndarray,
    centroids: list[Centroid],
    eps: float,
) -> Centroid | None:
    if len(centroids) == 0:
        return None
    
    min_dist = float('inf')
    nearest = None
    
    for centroid in centroids:
        dist = np.linalg.norm(vector - centroid.position)
        if dist < min_dist:
            min_dist = dist
            nearest = centroid
    
    if min_dist <= eps:
        return nearest
    
    return None


def split_centroid(centroid: Centroid, next_centroid_id: int, target_label: int) -> tuple[Centroid, Centroid, int]:
    """
    Centroid를 두 개로 분할
    
    방법: KMeans로 멤버를 2개 그룹으로 분할
    """
    members_array = np.stack(centroid.members, axis=0)
    
    if len(members_array) < 2:
        # 멤버가 2개 미만이면 분할 불가
        return centroid, None, next_centroid_id
    
    # KMeans로 2개 그룹으로 분할
    kmeans = KMeans(n_clusters=2, random_state=1, n_init=10)
    cluster_labels = kmeans.fit_predict(members_array)
    
    # 두 그룹으로 나누기
    group1_mask = (cluster_labels == 0)
    group2_mask = (cluster_labels == 1)
    
    group1_members = [centroid.members[i] for i in range(len(centroid.members)) if group1_mask[i]]
    group2_members = [centroid.members[i] for i in range(len(centroid.members)) if group2_mask[i]]
    
    # 각 그룹이 최소 1개 이상의 멤버를 가져야 함
    if len(group1_members) == 0 or len(group2_members) == 0:
        # 분할 실패 (중복 벡터 등으로 인해 KMeans가 제대로 분할하지 못함)
        return centroid, None, next_centroid_id
    
    # 기존 centroid를 첫 번째 그룹으로 업데이트
    centroid.members = group1_members
    centroid._recalculate_position()
    centroid._recalculate_radius()
    
    # 두 번째 그룹으로 새 centroid 생성
    group2_array = np.stack(group2_members, axis=0)
    group2_center = np.mean(group2_array, axis=0)
    
    new_centroid = Centroid(
        centroid_id=next_centroid_id,
        label=target_label,
        position=group2_center,
    )
    new_centroid.members = group2_members
    new_centroid._recalculate_radius()
    
    return centroid, new_centroid, next_centroid_id + 1


def incremental_clustering(
    vectors: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    eps: float,
    overlap_ratio_threshold: float = 0.3,
) -> list[Centroid]:
    mask = (labels == target_label)
    class_vectors = vectors[mask]
    
    print(f"[INFO] Class = {target_label} | n_samples = {class_vectors.shape[0]}")
    
    if class_vectors.shape[0] == 0:
        print(f"[WARN] No samples for class = {target_label}.")
        return []
    
    centroids: list[Centroid] = []
    next_centroid_id = 0
    
    if class_vectors.shape[0] > 0:
        first_centroid = Centroid(
            centroid_id=next_centroid_id,
            label=target_label,
            position=class_vectors[0].copy(),
        )
        centroids.append(first_centroid)
        next_centroid_id += 1
        print(f"[INFO] Created initial centroid {first_centroid.centroid_id} for label {target_label}")
    
    for i in range(1, class_vectors.shape[0]):
        vector = class_vectors[i]
        
        nearest = find_nearest_centroid(vector, centroids, eps)
        
        if nearest is not None:
            nearest.add_member(vector)
            
            # 겹치는 모든 centroid 찾기 (자기 자신 포함)
            overlapping = nearest.find_overlapping_centroids(centroids, overlap_ratio_threshold)
            
            if len(overlapping) > 0:
                # 겹치는 centroid들 중에서 멤버가 가장 많은 것을 분할
                overlapping.append(nearest)  # 자기 자신도 포함
                centroid_to_split = max(overlapping, key=lambda c: len(c.members))
                
                if len(centroid_to_split.members) >= 2:
                    updated_centroid, new_centroid, next_centroid_id = split_centroid(
                        centroid_to_split, next_centroid_id, target_label
                    )
                    
                    if new_centroid is not None:
                        # 새로 생성된 centroid는 이미 centroids에 추가되어 있지 않으므로 추가
                        if centroid_to_split.centroid_id == nearest.centroid_id:
                            # nearest를 분할한 경우, updated_centroid는 이미 centroids에 있음
                            centroids.append(new_centroid)
                        else:
                            # 다른 centroid를 분할한 경우
                            # 기존 centroid를 업데이트하고 새 centroid 추가
                            idx = next(i for i, c in enumerate(centroids) if c.centroid_id == centroid_to_split.centroid_id)
                            centroids[idx] = updated_centroid
                            centroids.append(new_centroid)
                        
                        print(f"[INFO] Split centroid {updated_centroid.centroid_id} -> created {new_centroid.centroid_id} "
                              f"(members: {len(updated_centroid.members)} + {len(new_centroid.members)})")
        else:
            new_centroid = Centroid(
                centroid_id=next_centroid_id,
                label=target_label,
                position=vector.copy(),
            )
            centroids.append(new_centroid)
            next_centroid_id += 1
        
        if (i + 1) % 1000 == 0:
            print(f"[INFO] Processed {i + 1}/{class_vectors.shape[0]} samples, "
                  f"centroids = {len(centroids)}")
    
    print(f"[INFO] Final centroids for label {target_label} = {len(centroids)}")
    return centroids


def main():
    
    numbering = find_numbering(RESULTS_DIR)
    result_path = os.path.join(RESULTS_DIR, numbering)
    os.makedirs(result_path, exist_ok=True)
    print(f"[INFO] Result path :: {result_path}")
    
    CENTROID_VECTORS_PATH = os.path.join(result_path, "centroid_vectors.npy")
    CENTROID_LABELS_PATH = os.path.join(result_path, "centroid_labels.npy")
    CENTROID_RADII_PATH = os.path.join(result_path, "centroid_radii.npy")
    CENTROID_COUNTS_PATH = os.path.join(result_path, "centroid_counts.npy")
    CENTROID_IDS_PATH = os.path.join(result_path, "centroid_ids.npy")
    
    GOOD_DB_PATH = os.path.join(result_path, "good_db_vectors.npy")
    BAD_DB_PATH = os.path.join(result_path, "bad_db_vectors.npy")
    
    # 통합 DB
    DB_VECTORS_PATH = os.path.join(result_path, "db_vectors.npy")
    DB_LABELS_PATH = os.path.join(result_path, "db_labels.npy")
    DB_COUNTS_PATH = os.path.join(result_path, "db_counts.npy")
    
    CONFIG_PATH = os.path.join(result_path, "config.json")
    TIME_PATH = os.path.join(result_path, "time.txt")
    
    # 1. 데이터 로드
    print(f"[INFO] Loading train dataset from {TRAIN_PATH} ...")
    X_train, y_train = load_dataset(TRAIN_PATH)
    print(f"[INFO] Train size :: {len(X_train)}")
    
    y_train = np.asarray(y_train, dtype=int)
    
    # 2. 임베딩 모델 로드
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[INFO] Loading embedding model :: {model_name}")
    embedder = Embedder(
        model_name=model_name,
        device=None,
    )
    print("[INFO] Embedding model loaded.")
    
    # 3. 전체 train 임베딩
    print("[INFO] Encoding train texts ...")
    t0 = time.perf_counter()
    train_vectors = embedder.encode(list(X_train))  # (N, D)
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Train embedding")
    print(f"[INFO] Train embedding done :: shape = {train_vectors.shape}")
    
    # 4. 클래스별 Incremental Clustering (0 = good, 1 = bad)
    print("[INFO] Running Incremental Clustering for GOOD (label = 0) ...")
    t0 = time.perf_counter()
    good_centroids = incremental_clustering(
        vectors=train_vectors,
        labels=y_train,
        target_label=0,
        eps=eps_good,
        overlap_ratio_threshold=overlap_ratio_threshold,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Incremental clustering (good)")
    
    print("[INFO] Running Incremental Clustering for BAD (label = 1) ...")
    t0 = time.perf_counter()
    bad_centroids = incremental_clustering(
        vectors=train_vectors,
        labels=y_train,
        target_label=1,
        eps=eps_bad,
        overlap_ratio_threshold=overlap_ratio_threshold,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Incremental clustering (bad)")
    
    # 5. Centroid 정보를 numpy 배열로 변환
    all_centroids = good_centroids + bad_centroids
    
    if len(all_centroids) == 0:
        raise RuntimeError("No centroids created!")
    
    centroid_vectors = np.stack([c.position for c in all_centroids], axis=0).astype(np.float32)
    centroid_labels = np.asarray([c.label for c in all_centroids], dtype=int)
    centroid_radii = np.asarray([c.radius for c in all_centroids], dtype=np.float32)
    centroid_counts = np.asarray([c.get_member_count() for c in all_centroids], dtype=int)
    centroid_ids = np.asarray([c.centroid_id for c in all_centroids], dtype=int)
    
    print(f"[INFO] Final centroids :: {len(all_centroids)} "
          f"(good = {len(good_centroids)}, bad = {len(bad_centroids)})")
    print(f"[INFO] Centroid vectors shape :: {centroid_vectors.shape}")
    
    # 6. Centroid 정보 저장
    np.save(CENTROID_VECTORS_PATH, centroid_vectors)
    np.save(CENTROID_LABELS_PATH, centroid_labels)
    np.save(CENTROID_RADII_PATH, centroid_radii)
    np.save(CENTROID_COUNTS_PATH, centroid_counts)
    np.save(CENTROID_IDS_PATH, centroid_ids)
    print(f"[INFO] Saved centroid vectors :: {CENTROID_VECTORS_PATH}")
    print(f"[INFO] Saved centroid labels  :: {CENTROID_LABELS_PATH}")
    print(f"[INFO] Saved centroid radii   :: {CENTROID_RADII_PATH}")
    print(f"[INFO] Saved centroid counts  :: {CENTROID_COUNTS_PATH}")
    print(f"[INFO] Saved centroid IDs     :: {CENTROID_IDS_PATH}")
    
    # 7. 기존 코드 호환용 good / bad 분리 벡터
    good_db_vectors = centroid_vectors[centroid_labels == 0]
    bad_db_vectors = centroid_vectors[centroid_labels == 1]
    
    np.save(GOOD_DB_PATH, good_db_vectors)
    np.save(BAD_DB_PATH, bad_db_vectors)
    print(f"[INFO] Saved GOOD DB vectors :: {GOOD_DB_PATH}")
    print(f"[INFO] Saved BAD  DB vectors :: {BAD_DB_PATH}")
    
    # 8. 통합 DB 저장
    db_vectors = centroid_vectors
    db_labels = centroid_labels
    db_counts = centroid_counts
    
    np.save(DB_VECTORS_PATH, db_vectors)
    np.save(DB_LABELS_PATH, db_labels)
    np.save(DB_COUNTS_PATH, db_counts)
    print(f"[INFO] Saved DB vectors :: {DB_VECTORS_PATH}")
    print(f"[INFO] Saved DB labels  :: {DB_LABELS_PATH}")
    print(f"[INFO] Saved DB counts  :: {DB_COUNTS_PATH}")
    
    # 9. config 저장
    config = {
        "dataset_language": dataset,
        "method": method,
        "embedder_model": model_name,
        "eps_good": eps_good,
        "eps_bad": eps_bad,
        "overlap_ratio_threshold": overlap_ratio_threshold,
        "centroid_count": int(len(all_centroids)),
        "centroid_count_good": int(len(good_centroids)),
        "centroid_count_bad": int(len(bad_centroids)),
        "db_dim": int(centroid_vectors.shape[1]),
    }
    write_config(config, CONFIG_PATH)
    print(f"[INFO] Saved config :: {CONFIG_PATH}")
    
    print("[INFO] 03_incremental DB build finished.")


if __name__ == "__main__":
    main()
