# Abstract base class for recommenders
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

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
        user_history = self.data_loader.get_user_history(user_id)
        return set(user_history['click_article_id'].tolist())

    def evaluate(self, test_data: pd.DataFrame) -> float:
        """
        Evaluate the recommender model on the test data.

        Args:
            test_data (pd.DataFrame): The test data containing user-item interactions.
        Returns:
            float: The evaluation metric score (e.g., precision, recall, etc.).
        """
        self.prepare_embeddings(test_data)

        # Instantiate lists to store evaluation metrics
        hits, precisions, recalls, f1s = [], [], [], []

        # Iterate over each user in the test data
        for user_id in tqdm(test_data["user_id"].unique()):
            ic = self.interaction_item_col
            true_items = set(
                test_data.loc[test_data["user_id"] == user_id, ic].unique()
            )
            # If a user has no true items, skip evaluation for this user
            if not true_items:
                continue

            recommended_items = self.recommend(user_id, num_recommendations=self.k)
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

        return {
            f"Hit@{self.k}": np.mean(hits).round(4),
            f"Precision@{self.k}": np.mean(precisions).round(4),
            f"Recall@{self.k}": np.mean(recalls).round(4),
            f"F1@{self.k}": np.mean(f1s).round(4),
        }

    def _format_recommendation(self, article_id: int, score: float, reason: str = "") -> Dict[str, Any]:
        """Format a recommendation"""
        article_info = self.data_loader.get_article_info(article_id)

        return {
            "article_id": int(article_id),
            "score": float(score),
            "reason": reason,
            "metadata": article_info
        }
