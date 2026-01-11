from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import pairwise_distances_chunked


def build_knn_features (
    query_vectors : np.ndarray,
    good_db_vectors : np.ndarray,
    bad_db_vectors : np.ndarray,
    k : int = 49,
) -> np.ndarray :

    if (good_db_vectors is None or len(good_db_vectors) == 0) or (bad_db_vectors is None or len(bad_db_vectors) == 0) :
        raise ValueError("GOOD / BAD DB are empty. Please check good_db_vectors and bad_db_vectors.")

    db_list = []
    label_list = []

    if (good_db_vectors is not None) and (len(good_db_vectors) > 0) :
        db_list.append(good_db_vectors)
        label_list.append(np.zeros(len(good_db_vectors), dtype=int))

    if (bad_db_vectors is not None) and (len(bad_db_vectors) > 0) :
        db_list.append(bad_db_vectors)
        label_list.append(np.ones(len(bad_db_vectors), dtype=int))

    db_vectors = np.concatenate(db_list, axis=0)       # (M, D)
    db_labels = np.concatenate(label_list, axis=0)     # (M,)

    N = query_vectors.shape[0]
    M = db_vectors.shape[0]

    if (M == 0) :
        raise ValueError("Integrated DB vectors are empty.")

    k_eff = min(k, M)

    # 메모리 효율적인 배치 처리
    topk_dists_list = []
    topk_labels_list = []
    
    gen = pairwise_distances_chunked(
        query_vectors,
        db_vectors,
        metric='euclidean',
        n_jobs=1,
        working_memory=1024,
    )
    
    for dists_chunk in gen:
        chunk_size = dists_chunk.shape[0]
        
        # 각 chunk에 대해 top-k 찾기
        idx_part = np.argpartition(dists_chunk, k_eff - 1, axis=1)[:, :k_eff]
        row_indices = np.arange(chunk_size)[:, None]
        dists_part = dists_chunk[row_indices, idx_part]
        order = np.argsort(dists_part, axis=1)
        
        topk_idx = idx_part[row_indices, order]
        topk_dists_chunk = dists_chunk[row_indices, topk_idx]
        topk_labels_chunk = db_labels[topk_idx]
        
        topk_dists_list.append(topk_dists_chunk)
        topk_labels_list.append(topk_labels_chunk)
    
    topk_dists = np.concatenate(topk_dists_list, axis=0)
    topk_labels = np.concatenate(topk_labels_list, axis=0)

    if k_eff < k:
        pad_width = k - k_eff
        pad_d = np.full((N, pad_width), np.nan, dtype=float)
        pad_l = np.full((N, pad_width), np.nan, dtype=float)
        topk_dists = np.concatenate([topk_dists, pad_d], axis=1)
        topk_labels = np.concatenate([topk_labels.astype(float), pad_l], axis=1)
    else:
        topk_labels = topk_labels.astype(float)

    feats = np.concatenate([topk_dists, topk_labels], axis=1)  # (N, 2*k)
    return feats


@dataclass
class DistanceNNConfig :
    input_dim : int
    hidden_dim1 : int = 64
    hidden_dim2 : int = 32
    dropout : float = 0.0


class DistanceNN(nn.Module) :

    def __init__(self, config: DistanceNNConfig) :
        super().__init__()
        layers = []

        layers.append(nn.Linear(config.input_dim, config.hidden_dim1))
        layers.append(nn.ReLU())
        if (config.dropout > 0.0) :
            layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(config.hidden_dim1, config.hidden_dim2))
        layers.append(nn.ReLU())
        if (config.dropout > 0.0) :
            layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(config.hidden_dim2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        out = self.net(x)
        return out.squeeze(-1)