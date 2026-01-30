from __future__ import annotations

import os
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.core.data import load_dataset, find_numbering
from src.core.embedding import Embedder
from src.core.metrics import write_time, write_config


# ===== config =====
dataset = "en"

# Incremental clustering hyper-parameters
eps_1_good = 0.5  # distance threshold for good
eps_1_bad = 0.5   # distance threshold for bad

eps_2_good = 0.9  # overlap ratio threshold for good
eps_2_bad = 0.9    # overlap ratio threshold for bad

method = "04_incremental"
TRAIN_PATH = os.path.join("dataset", dataset, "train_all.csv")

RESULTS_DIR = os.path.join("results", dataset, "experiment05_clustering", method)


class Centroid:

    def __init__(self, centroid_id: int, label: int, position: np.ndarray, initial_radius: float):
        self.centroid_id = centroid_id
        self.label = label
        self.position = position.copy()
        self.members = [position.copy()]
        self.radius = initial_radius
    
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
    
    def find_overlapping_centroid(self, other_centroids: list['Centroid'], eps_2: float) -> 'Centroid' | None:
        """
        같은 label의 다른 centroid와 겹치는지 확인
        겹침 조건: (distance - max(r1, r2)) <= min(r1, r2) * eps_2
        
        Returns:
            겹치는 centroid가 있으면 그 중 하나 반환, 없으면 None
        """
        if len(self.members) < 1:
            return None
        
        for other in other_centroids:
            # 자기 자신이거나 다른 label이면 스킵
            if other.centroid_id == self.centroid_id or other.label != self.label:
                continue
            
            distance = np.linalg.norm(self.position - other.position)
            r1, r2 = self.radius, other.radius
            
            # 겹침 조건: (distance - max(r1, r2)) <= min(r1, r2) * eps_2
            min_radius = min(r1, r2)
            max_radius = max(r1, r2)
            
            if (distance - max_radius) <= min_radius * eps_2:
                return other
        
        return None


def find_nearest_centroid(
    vector: np.ndarray,
    centroids: list[Centroid],
    eps: float,
    target_label: int,
) -> Centroid | None:
    """
    같은 label의 centroid 중에서 가장 가까운 것을 찾음
    거리가 eps 이하인 경우에만 반환
    """
    if len(centroids) == 0:
        return None
    
    min_dist = float('inf')
    nearest = None
    
    for centroid in centroids:
        # 같은 label만 체크
        if centroid.label != target_label:
            continue
        
        dist = np.linalg.norm(vector - centroid.position)
        if dist < min_dist:
            min_dist = dist
            nearest = centroid
    
    if min_dist <= eps:
        return nearest
    
    return None


def determine_k_by_silhouette(members_array: np.ndarray) -> int:
    """
    k=2와 k=3 중에서 silhouette score가 더 높은 k를 선택
    """
    if len(members_array) < 3:
        return 2  # 멤버가 3개 미만이면 k=2
    
    scores = {}
    
    for k in [2, 3]:
        if len(members_array) < k:
            continue
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
            cluster_labels = kmeans.fit_predict(members_array)
            
            # silhouette score 계산
            if len(np.unique(cluster_labels)) < 2:
                # 클러스터가 1개만 있으면 silhouette score 계산 불가
                continue
            
            score = silhouette_score(members_array, cluster_labels)
            scores[k] = score
        except Exception:
            continue
    
    if len(scores) == 0:
        return 2  # 기본값
    
    # silhouette score가 가장 높은 k 선택
    best_k = max(scores, key=scores.get)
    return best_k


def merge_and_split_centroids(
    c_i: Centroid,
    c_j: Centroid,
    next_centroid_id: int,
    target_label: int,
    eps_1: float,
) -> list[Centroid]:
    """
    두 centroid를 병합하고 KMeans로 재분할
    
    Returns:
        재분할된 centroid 리스트
    """
    # 두 centroid의 멤버 합치기
    all_members = c_i.members + c_j.members
    
    if len(all_members) < 2:
        # 멤버가 2개 미만이면 분할 불가
        return [c_i, c_j]
    
    members_array = np.stack(all_members, axis=0)
    
    # silhouette score로 최적 k 결정
    k = determine_k_by_silhouette(members_array)
    
    # KMeans로 분할
    try:
        kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
        cluster_labels = kmeans.fit_predict(members_array)
    except Exception:
        # KMeans 실패 시 원래대로 반환
        return [c_i, c_j]
    
    # 각 클러스터로 centroid 생성
    new_centroids = []
    
    for cluster_id in range(k):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_members = [all_members[i] for i in range(len(all_members)) if cluster_mask[i]]
        
        if len(cluster_members) == 0:
            continue
        
        cluster_array = np.stack(cluster_members, axis=0)
        cluster_center = np.mean(cluster_array, axis=0)
        
        new_centroid = Centroid(
            centroid_id=next_centroid_id,
            label=target_label,
            position=cluster_center,
            initial_radius=eps_1,
        )
        new_centroid.members = cluster_members
        new_centroid._recalculate_radius()
        
        new_centroids.append(new_centroid)
        next_centroid_id += 1
    
    return new_centroids


def incremental_clustering(
    vectors: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    eps_1: float,
    eps_2: float,
) -> list[Centroid]:
    """
    Incremental clustering 수행
    """
    mask = (labels == target_label)
    class_vectors = vectors[mask]
    
    print(f"[INFO] Class = {target_label} | n_samples = {class_vectors.shape[0]}")
    
    if class_vectors.shape[0] == 0:
        print(f"[WARN] No samples for class = {target_label}.")
        return []
    
    centroids: list[Centroid] = []
    next_centroid_id = 0
    
    for i in range(class_vectors.shape[0]):
        vector = class_vectors[i]
        
        # 같은 label의 centroid 중에서 가장 가까운 것 찾기
        nearest = find_nearest_centroid(vector, centroids, eps_1, target_label)
        
        if nearest is not None:
            # 가장 가까운 centroid에 멤버 추가
            nearest.add_member(vector)
            
            # 겹치는 centroid 찾기
            overlapping = nearest.find_overlapping_centroid(centroids, eps_2)
            
            if overlapping is not None:
                # 두 centroid의 멤버 합이 3개 이상이면 병합 및 재분할
                total_members = len(nearest.members) + len(overlapping.members)
                
                if total_members >= 3:
                    # 두 centroid를 리스트에서 제거
                    centroids = [c for c in centroids if c.centroid_id not in [nearest.centroid_id, overlapping.centroid_id]]
                    
                    # 병합 및 재분할
                    new_centroids = merge_and_split_centroids(
                        nearest, overlapping, next_centroid_id, target_label, eps_1
                    )
                    
                    # 새 centroid들 추가
                    centroids.extend(new_centroids)
                    next_centroid_id += len(new_centroids)
                    
                    print(f"[INFO] Merged and split centroids -> created {len(new_centroids)} new centroids")
        else:
            # 새로운 centroid 생성
            new_centroid = Centroid(
                centroid_id=next_centroid_id,
                label=target_label,
                position=vector.copy(),
                initial_radius=eps_1,
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
        eps_1=eps_1_good,
        eps_2=eps_2_good,
    )
    t1 = time.perf_counter()
    write_time(t1 - t0, TIME_PATH, "Incremental clustering (good)")
    
    print("[INFO] Running Incremental Clustering for BAD (label = 1) ...")
    t0 = time.perf_counter()
    bad_centroids = incremental_clustering(
        vectors=train_vectors,
        labels=y_train,
        target_label=1,
        eps_1=eps_1_bad,
        eps_2=eps_2_bad,
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
        "eps_1_good": eps_1_good,
        "eps_1_bad": eps_1_bad,
        "eps_2_good": eps_2_good,
        "eps_2_bad": eps_2_bad,
        "centroid_count": int(len(all_centroids)),
        "centroid_count_good": int(len(good_centroids)),
        "centroid_count_bad": int(len(bad_centroids)),
        "db_dim": int(centroid_vectors.shape[1]),
    }
    write_config(config, CONFIG_PATH)
    print(f"[INFO] Saved config :: {CONFIG_PATH}")
    
    print("[INFO] 04_incremental DB build finished.")


if __name__ == "__main__":
    main()
