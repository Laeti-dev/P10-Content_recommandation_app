# Abstract base class for recommenders
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Recommender(ABC):
    MODEL_NAME = "base"

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    @abstractmethod
    def recommend(self, user_id: int, num_recommendations: int = 10) -> list[int]:
        pass


class ContentBasedRecommender(Recommender):
    """Content-based recommender using a user profile vector and cosine similarity to items."""

    MODEL_NAME = "content-based"

    def __init__(
        self,
        item_embeddings: Mapping[int, np.ndarray],
        user_item_history: Mapping[int, Sequence[int]],
        *,
        recency_decay: float | None = None,
    ) -> None:
        """
        Args:
            item_embeddings: Maps each item id to its embedding vector.
            user_item_history: Maps each user id to clicked item ids (order = chronological).
            recency_decay: If set (e.g. 0.9), weights recent clicks more when building the profile.
        """
        self._item_embeddings = {
            int(k): np.asarray(v, dtype=np.float64).ravel()
            for k, v in item_embeddings.items()
        }
        self._user_history = {
            int(uid): [int(i) for i in items] for uid, items in user_item_history.items()
        }
        if recency_decay is not None and not (0 < recency_decay <= 1):
            raise ValueError("recency_decay must be in (0, 1] or None")
        self._recency_decay = recency_decay

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        history = self._user_history.get(user_id)
        if not history:
            return None

        vectors: list[np.ndarray] = []
        weights: list[float] = []
        # Iterate from most recent click to oldest so decay applies to recency.
        for rank, item_id in enumerate(reversed(history)):
            vec = self._item_embeddings.get(item_id)
            if vec is None or vec.size == 0:
                continue
            w = 1.0 if self._recency_decay is None else self._recency_decay**rank
            vectors.append(vec)
            weights.append(w)

        if not vectors:
            return None

        w_arr = np.asarray(weights, dtype=np.float64)
        w_arr /= w_arr.sum()
        stacked = np.stack(vectors, axis=0)
        return np.average(stacked, axis=0, weights=w_arr)

    def recommend(self, user_id: int, num_recommendations: int = 10) -> list[int]:
        profile = self._user_profile(user_id)
        if profile is None:
            return []

        seen = set(self._user_history.get(user_id, []))
        candidates = [iid for iid in self._item_embeddings if iid not in seen]
        if not candidates:
            return []

        matrix = np.stack([self._item_embeddings[iid] for iid in candidates])
        sims = cosine_similarity(profile.reshape(1, -1), matrix, dense_output=True)[0]
        top_idx = np.argsort(-sims)[:num_recommendations]
        return [candidates[int(i)] for i in top_idx]


class CollaborativeFilteringRecommender(Recommender):
    MODEL_NAME = "collaborative-filtering"

    def recommend(self, user_id: int, num_recommendations: int = 10) -> list[int]:
        return [4, 5, 6]
