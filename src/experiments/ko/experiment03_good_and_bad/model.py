from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional

import numpy as np

from src.core.embedding import Embedder


@dataclass
class GoodBadKNNClassifier :
    
    seed_size : int = 20
    k : int = 1
    random_state : int = 1
    
    good_vectors : Optional[np.ndarray] = field(default = None, init = False)
    bad_vectors  : Optional[np.ndarray] = field(default = None, init = False)
    good_texts   : List[str] = field(default_factory = list, init = False)
    bad_texts    : List[str] = field(default_factory = list, init = False)
    
    def _init_db (
        self,
        X_train : Sequence[str],
        vectors : np.ndarray,
        y_train : np.ndarray,
    ) -> int :

        n_seed = min(self.seed_size, len(X_train))
        
        good_list = []
        bad_list  = []
        
        for i in range(n_seed) :
            vec   = vectors[i]
            text  = X_train[i]
            label = y_train[i]
            
            if label == 1 :   # bad
                self.bad_texts.append(text)
                bad_list.append(vec)
            else :            # good
                self.good_texts.append(text)
                good_list.append(vec)
        
        self.good_vectors = np.stack(good_list, axis = 0) if good_list else None
        self.bad_vectors  = np.stack(bad_list,  axis = 0) if bad_list  else None
        
        return n_seed
    
    def _predict_vector (self, vec : np.ndarray) -> int :
        
        if (self.good_vectors is None) and (self.bad_vectors is None) :
            raise RuntimeError("GOOD DB와 BAD DB가 모두 비어 있습니다. fit()을 먼저 호출하세요.")
        
        dists_all  : List[float] = []
        labels_all : List[int]   = []
        
        # good DB 거리
        if (self.good_vectors is not None) and (len(self.good_vectors) > 0) :
            dists_good = np.linalg.norm(self.good_vectors - vec, axis = 1)
            dists_all.extend(dists_good.tolist())
            labels_all.extend([0] * len(dists_good))  # good = 0
        
        # bad DB 거리
        if (self.bad_vectors is not None) and (len(self.bad_vectors) > 0) :
            dists_bad = np.linalg.norm(self.bad_vectors - vec, axis = 1)
            dists_all.extend(dists_bad.tolist())
            labels_all.extend([1] * len(dists_bad))   # bad = 1
        
        if not dists_all :
            raise RuntimeError("GOOD / BAD DB에 유효한 벡터가 없습니다.")
        
        dists_all  = np.asarray(dists_all, dtype = float)
        labels_all = np.asarray(labels_all, dtype = int)
        
        k_eff = min(self.k, len(dists_all))
        
        # 가장 가까운 k개 이웃 선택
        idx = np.argpartition(dists_all, k_eff - 1)[:k_eff]
        neighbor_labels = labels_all[idx]
        
        # 다수결 (0 = good, 1 = bad), 동률이면 bad(1) 쪽으로
        mean_label = neighbor_labels.mean()
        return 1 if mean_label >= 0.5 else 0
    
    def fit (
        self,
        X_train : Sequence[str],
        y_train : Sequence[int],
        embedder : Embedder,
    ) -> "GoodBadKNNClassifier" :
        
        if len(X_train) != len(y_train) :
            raise ValueError("X_train과 y_train의 길이가 다릅니다.")
        
        X_train = list(X_train)
        y_train = np.array(y_train, dtype = int)
        
        print("[INFO] Encoding train texts ...")
        vectors = embedder.encode(X_train)  # (N, D)
        
        # 1) seed로 GOOD / BAD DB 초기화
        print(f"[INFO] Initializing GOOD/BAD DB with first {self.seed_size} samples ...")
        start_idx = self._init_db(X_train, vectors, y_train)
        n_samples = len(X_train)
        
        print("[INFO] Start online training with remaining samples ...")
        for i in range(start_idx, n_samples) :
            vec   = vectors[i]
            text  = X_train[i]
            y_true = y_train[i]
            
            y_pred = self._predict_vector(vec)
            
            if y_pred == y_true :
                continue
            
            # 오분류된 경우 정답 라벨 DB에 추가
            if y_true == 1 :
                # BAD DB에 추가
                if self.bad_vectors is None :
                    self.bad_vectors = vec[None, :]
                else :
                    self.bad_vectors = np.vstack([self.bad_vectors, vec])
                self.bad_texts.append(text)
            else :
                # GOOD DB에 추가
                if self.good_vectors is None :
                    self.good_vectors = vec[None, :]
                else :
                    self.good_vectors = np.vstack([self.good_vectors, vec])
                self.good_texts.append(text)
        
        print("[INFO] Training finished.")
        print(f"[INFO] GOOD DB size: {0 if self.good_vectors is None else len(self.good_vectors)}")
        print(f"[INFO] BAD  DB size: {0 if self.bad_vectors  is None else len(self.bad_vectors)}")
        
        return self
    
    def predict (
        self,
        X : Sequence[str],
        embedder : Embedder,
    ) -> np.ndarray :
        
        X = list(X)
        vectors = embedder.encode(X)
        
        preds : List[int] = []
        
        for vec in vectors :
            preds.append(self._predict_vector(vec))
        
        return np.array(preds, dtype = int)
    
    def predict_with_distances (
        self,
        X : Sequence[str],
        embedder : Embedder,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
        
        X = list(X)
        vectors = embedder.encode(X)
        
        if (self.good_vectors is None) and (self.bad_vectors is None) :
            raise RuntimeError("GOOD / BAD DB가 모두 비어 있습니다. fit()을 먼저 호출하세요.")
        
        y_pred_list : List[int]   = []
        d_good_list : List[float] = []
        d_bad_list  : List[float] = []
        
        for vec in vectors :
            # good
            if (self.good_vectors is not None) and (len(self.good_vectors) > 0) :
                dists_good = np.linalg.norm(self.good_vectors - vec, axis = 1)
                d_good = float(dists_good.min())
            else :
                d_good = float("inf")
            
            # bad
            if (self.bad_vectors is not None) and (len(self.bad_vectors) > 0) :
                dists_bad = np.linalg.norm(self.bad_vectors - vec, axis = 1)
                d_bad = float(dists_bad.min())
            else :
                d_bad = float("inf")
            
            if (not np.isfinite(d_good)) and (not np.isfinite(d_bad)) :
                raise RuntimeError("GOOD / BAD DB가 모두 비어 있습니다.")
            
            if d_bad < d_good :
                y_pred = 1
            else :
                y_pred = 0
            
            y_pred_list.append(y_pred)
            d_good_list.append(d_good)
            d_bad_list.append(d_bad)
        
        return (
            np.array(y_pred_list, dtype = int),
            np.array(d_good_list, dtype = float),
            np.array(d_bad_list,  dtype = float),
        )
