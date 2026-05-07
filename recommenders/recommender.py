# Abstract base class for recommenders
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
import time

import numpy as np
import logging


logger = logging.getLogger(__name__)

class Recommender(ABC):
    def __init__(
        self,
        data_loader=None,
        k: int = 5,
        *,
        interaction_item_col: str = "click_article_id",
    ):
        self.data_loader = data_loader
        self.name = self.__class__.__name__
        self.k = k
        self.interaction_item_col = interaction_item_col

    @abstractmethod
    def recommend(self, user_id: int, num_recommendations: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Recommend items for a given user."""
        raise NotImplementedError("Subclasses must implement this method")

    def prepare_embeddings(self, test_data: pd.DataFrame) -> None:
        """Optional hook before evaluation (e.g. load and restrict embedding table)."""
        pass

    def _get_user_clicked_items(self, user_id: int) -> set:
        """Get the items that the user has already seen."""
        if self.data_loader is None:
            raise ValueError("data_loader is required to fetch user history")
        user_history = self.data_loader.get_user_history(user_id)
        ic = self.interaction_item_col
        if ic not in user_history.columns:
            raise ValueError(
                f"user history missing expected item column '{ic}'. "
                f"Available columns: {sorted(user_history.columns.tolist())}"
            )
        return set(user_history[ic].astype(int).tolist())

    def get_item_embeddings(self, article_ids: List[int]) -> np.ndarray:
        """
        Return embedding rows for article_ids using the configured data_loader.

        The project DataLoader loads a pickled numpy matrix where rows are indexed by
        article_id (0..n-1) for this dataset.
        """
        if self.data_loader is None:
            raise ValueError("data_loader is required to fetch embeddings")
        return self.data_loader.get_embeddings_by_ids([int(x) for x in article_ids])

    def build_user_profile(
        self,
        user_id: int,
        interactions_df: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: Optional[str] = None,
        weight_col: Optional[str] = "interaction_weight",
    ) -> np.ndarray:
        """
        Build a user profile vector by aggregating embeddings of interacted items.

        If ``weight_col`` exists in interactions_df, uses a weighted average; otherwise
        uses a simple mean.
        """
        ic = item_col or self.interaction_item_col
        if user_col not in interactions_df.columns:
            raise ValueError(f"interactions_df missing '{user_col}' column")
        if ic not in interactions_df.columns:
            raise ValueError(f"interactions_df missing '{ic}' column")

        user_rows = interactions_df.loc[interactions_df[user_col] == user_id]
        if len(user_rows) == 0:
            raise ValueError(f"no interactions found for user_id={user_id}")

        article_ids = user_rows[ic].astype(int).tolist()
        X = self.get_item_embeddings(article_ids)
        if X.size == 0:
            raise ValueError(f"no valid embeddings found for user_id={user_id}")

        if weight_col and weight_col in user_rows.columns:
            w = user_rows[weight_col].astype(float).to_numpy()
            w_sum = float(w.sum())
            if w_sum > 0:
                return (X * w[:, None]).sum(axis=0) / w_sum

        return X.mean(axis=0)

    def build_users_profiles(
        self,
        interactions_df: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: Optional[str] = None,
        weight_col: Optional[str] = "interaction_weight",
    ) -> Dict[int, np.ndarray]:
        """Build user profiles for every user present in interactions_df."""
        if user_col not in interactions_df.columns:
            raise ValueError(f"interactions_df missing '{user_col}' column")

        profiles: Dict[int, np.ndarray] = {}
        for uid in interactions_df[user_col].dropna().unique().tolist():
            user_id = int(uid)
            profiles[user_id] = self.build_user_profile(
                user_id,
                interactions_df,
                user_col=user_col,
                item_col=item_col,
                weight_col=weight_col,
            )
        return profiles


    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the recommender model on the test data.

        Args:
            test_data (pd.DataFrame): The test data containing user-item interactions.
        Returns:
            float: The evaluation metric score (e.g., precision, recall, etc.).
        """
        self.prepare_embeddings(test_data)

        # Instantiate lists to store evaluation metrics
        hits, precisions, recalls, f1s, latencies = [], [], [], [], []

        # Iterate over each user in the test data
        for user_id in tqdm(test_data["user_id"].unique()):
            ic = self.interaction_item_col
            true_items = set(
                test_data.loc[test_data["user_id"] == user_id, ic].unique()
            )
            # If a user has no true items, skip evaluation for this user
            if not true_items:
                continue
            start_time = time.perf_counter()
            recommended_items = self.recommend(user_id, num_recommendations=self.k)
            end_time = time.perf_counter()
            logger.info(f"Time taken to recommend items for user {user_id}: {end_time - start_time} seconds")

            rec_ids = [
                int(item["article_id"]) if isinstance(item, dict) else int(item)
                for item in recommended_items
            ]
            top_k = set(rec_ids[: self.k])
            n_hit = len(true_items & top_k)

            hit = 1.0 if n_hit > 0 else 0.0
            precision = n_hit / len(top_k)
            recall = n_hit / len(true_items)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            hits.append(hit)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            latencies.append(end_time - start_time)

        return {
            f"Hit@{self.k}": np.mean(hits).round(4),
            f"Precision@{self.k}": np.mean(precisions).round(4),
            f"Recall@{self.k}": np.mean(recalls).round(4),
            f"F1@{self.k}": np.mean(f1s).round(4),
            "Avg Latency (ms)": (np.mean(latencies) * 1000).round(2),
        }

    def _format_recommendation(self, article_id: int, score: float, reason: str = "") -> Dict[str, Any]:
        """Format a recommendation"""
        if self.data_loader is None:
            raise ValueError("data_loader is required to fetch article metadata")
        article_info = self.data_loader.get_article_info(article_id)

        return {
            "article_id": int(article_id),
            "score": float(score),
            "reason": reason,
            "metadata": article_info
        }
