from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional
import numpy as np

from src.core.embedding import Embedder

@dataclass
class AdaptiveBadDBClassifier : 
    
    initial_threshold : float = 1.0 
    min_threshold : float = 0.1
    max_threshold : float = 1.0
    threshold_step : float = 0.005
    
    seed_size : int = 10
    random_state : int = 1

    threshold_history : List[float] = field(default_factory = list, init = False)
    bad_vectors : Optional[np.ndarray] = field(default = None, init = False)
    bad_texts : List[str] = field(default_factory = list, init = False)
    threshold_ : float = field(default = 0, init = False)
    
    def _init_bad_db (self, X_train : Sequence[str], vectors : np.ndarray, y_train : np.ndarray) -> np.ndarray : 
        
        pos_indices = np.where(y_train == 1)[0]
        
        rng = np.random.RandomState(self.random_state)
        seed_size = min(self.seed_size, len(pos_indices))
        seed_indices = rng.choice(pos_indices, size = seed_size, replace = False)
        
        self.bad_texts = [X_train[i] for i in seed_indices]
        self.bad_vectors = vectors[seed_indices]
        
        return seed_indices
    
    
    def fit (
        self,
        X_train: Sequence[str],
        y_train: Sequence[int],
        embedder: Embedder,
    ) -> "AdaptiveBadDBClassifier" :

        if len(X_train) != len(y_train) :
            raise ValueError("X_train and y_train have different lengths.")

        y_train = np.array(y_train, dtype=int)

        vectors = embedder.encode(list(X_train))

        self.threshold_ = float(self.initial_threshold)
        self.threshold_history = [self.threshold_]

        seed_indices = set(self._init_bad_db(X_train, vectors, y_train))

        n_samples = len(X_train)

        for i in range(n_samples):
            if i in seed_indices:
                continue

            vec = vectors[i]
            y_true = y_train[i]

            if self.bad_vectors is None or len(self.bad_vectors) == 0:
                continue

            dists = np.linalg.norm(self.bad_vectors - vec, axis=1)
            d_min = float(dists.min())

            y_pred = 1 if d_min < self.threshold_ else 0

            if y_true == 1 and y_pred == 0:
                self.bad_vectors = np.vstack([self.bad_vectors, vec])
                self.bad_texts.append(X_train[i])
                self.threshold_ += self.threshold_step

            elif y_true == 0 and y_pred == 1:
                self.threshold_ -= self.threshold_step

            self.threshold_ = max(self.min_threshold, min(self.max_threshold, self.threshold_))

            self.threshold_history.append(self.threshold_)
            
        return self

    def _predict_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if self.bad_vectors is None or len(self.bad_vectors) == 0:
            raise RuntimeError("BAD DB is empty. Please call fit first.")

        y_pred = []
        for vec in vectors:
            dists = np.linalg.norm(self.bad_vectors - vec, axis=1)
            d_min = float(dists.min())
            label = 1 if d_min < self.threshold_ else 0
            y_pred.append(label)
        return np.array(y_pred, dtype=int)

    def predict(
        self,
        X: Sequence[str],
        embedder: Embedder,
    ) -> np.ndarray : 
        vectors = embedder.encode(list(X))
        return self._predict_vectors(vectors)

    def predict_with_distance(
        self,
        X: Sequence[str],
        embedder: Embedder,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vectors = embedder.encode(list(X))

        if self.bad_vectors is None or len(self.bad_vectors) == 0:
            raise RuntimeError("BAD DB is empty. Please call fit first.")

        y_pred = []
        distances = []

        for vec in vectors:
            dists = np.linalg.norm(self.bad_vectors - vec, axis=1)
            d_min = float(dists.min())
            distances.append(d_min)

            label = 1 if d_min < self.threshold_ else 0
            y_pred.append(label)

        return np.array(y_pred, dtype=int), np.array(distances, dtype=float)

    def get_threshold_history(self) -> List[float]:
        return list(self.threshold_history) 