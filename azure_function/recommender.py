"""
Production CBRecommender for Azure Function.

Lightweight inference-only version — no fit(), no data_loader.
Loads pre-computed artifacts from Azure Blob Storage and serves recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class ProductionRecommender:
    """
    Inference-only recommender.
    Loads artifacts from Azure Blob Storage and serves top-K recommendations.

    Artifacts expected:
        candidate_ids.pkl         : np.ndarray of article IDs
        candidate_embeddings.pkl  : np.ndarray (n_articles x dim)
        candidate_norms.pkl       : np.ndarray (n_articles,)
        article_to_idx.pkl        : Dict[int, int]
        user_profiles.pkl         : Dict[int, np.ndarray]
        user_seen.pkl             : Dict[int, Set[int]]
        popularity_scores.pkl     : Dict[int, float]
    """

    def __init__(self):
        self.candidate_ids: Optional[np.ndarray] = None
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.candidate_norms: Optional[np.ndarray] = None
        self.article_to_idx: Dict[int, int] = {}
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_seen: Dict[int, Set[int]] = {}
        self.popularity_scores: Dict[int, float] = {}
        self._pop_vector: Optional[np.ndarray] = None  # cached at load time
        self._is_ready: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, artifacts: Dict[str, object]) -> "ProductionRecommender":
        """
        Populate recommender from a dict of pre-loaded artifacts.

        Args:
            artifacts: dict with keys matching artifact filenames (without .pkl).

        Returns:
            self (for chaining)
        """
        self.candidate_ids       = artifacts["candidate_ids"]
        self.candidate_embeddings = artifacts["candidate_embeddings"]
        self.candidate_norms     = artifacts["candidate_norms"]
        self.article_to_idx      = artifacts["article_to_idx"]
        self.user_profiles       = artifacts["user_profiles"]
        self.user_seen           = artifacts["user_seen"]
        self.popularity_scores   = artifacts["popularity_scores"]

        # Pre-compute popularity vector aligned with candidate_ids (done once)
        self._pop_vector = np.array(
            [self.popularity_scores.get(int(aid), 0.0) for aid in self.candidate_ids],
            dtype=np.float32,
        )

        # Ensure norms have no zero (safety)
        self.candidate_norms = np.where(
            self.candidate_norms == 0, 1e-12, self.candidate_norms
        )

        self._is_ready = True
        logger.info(
            "ProductionRecommender ready: %d articles, %d user profiles.",
            len(self.candidate_ids),
            len(self.user_profiles),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        topk: int = 5,
        beta: float = 0.8,
    ) -> List[int]:
        """
        Return top-K article IDs for a given user.

        Args:
            user_id : User ID (must exist in user_profiles).
            topk    : Number of recommendations to return.
            beta    : Popularity weight (0 = CB only, 1 = popularity only).

        Returns:
            List of article IDs sorted by descending score.

        Raises:
            ValueError: If user not found or recommender not loaded.
        """
        if not self._is_ready:
            raise ValueError("Recommender not loaded. Call load() first.")

        profile = self.user_profiles.get(int(user_id))
        if profile is None:
            raise ValueError(f"User {user_id} not found in user profiles.")

        u_norm = np.linalg.norm(profile)
        if u_norm == 0:
            raise ValueError(f"User {user_id} has an empty profile.")

        # --- CB score : cosine similarity ---
        cb_scores = (self.candidate_embeddings @ profile) / (self.candidate_norms * u_norm)

        # --- Combined score : CB + popularity ---
        if beta > 0.0 and self._pop_vector is not None:
            scores = (1.0 - beta) * cb_scores + beta * self._pop_vector
        else:
            scores = cb_scores

        # --- Mask already seen articles ---
        seen = self.user_seen.get(int(user_id), set())
        seen_indices = [
            self.article_to_idx[aid]
            for aid in seen
            if aid in self.article_to_idx
        ]
        if seen_indices:
            scores[seen_indices] = -np.inf

        # --- Top-K ---
        if len(scores) > topk:
            top_indices = np.argpartition(scores, -topk)[-topk:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:topk]

        return self.candidate_ids[top_indices].tolist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def user_exists(self, user_id: int) -> bool:
        return int(user_id) in self.user_profiles

    def n_articles(self) -> int:
        return len(self.candidate_ids) if self.candidate_ids is not None else 0

    def n_users(self) -> int:
        return len(self.user_profiles)
