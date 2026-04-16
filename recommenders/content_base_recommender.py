from __future__ import annotations
from recommenders.recommender import Recommender
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ContentBasedRecommender(Recommender):
    def __init__(
        self,
        data_loader,
        train_data: pd.DataFrame,
        *,
        k: int = 5,
        item_col: str = "click_article_id",
        user_history_limit: Optional[int] = 20  # On limite aux 20 derniers articles
    ) -> None:
        super().__init__(data_loader, k=k, interaction_item_col=item_col)
        self.train_data = train_data
        self._item_col = item_col
        self.user_history_limit = user_history_limit
        self._embeddings_df: pd.DataFrame | None = None
        self._valid_article_ids: set[int] | None = None

    def _get_user_profile(self, user_id: int) -> np.ndarray:
        """Calcule le profil utilisateur vectorisé (Moyenne des derniers clics)."""
        history = self.data_loader.get_user_history(user_id, limit=self.user_history_limit)
        if history.empty:
            return None

        # Unique articles pour éviter de biaser le profil par des clics multiples
        user_article_ids = history[self._item_col].unique().tolist()

        # Extraction massive via la nouvelle méthode du data_loader
        user_vecs = self.data_loader.get_embeddings_by_ids(user_article_ids)

        if len(user_vecs) == 0:
            return None

        return np.mean(user_vecs, axis=0).reshape(1, -1)

    def _recommend(
        self, user_id: int, num_recommendations: int, **kwargs
    ) -> List[Dict[str, Any]]:
        # 1. Obtenir le profil
        user_profile = self._get_user_profile(user_id)
        if user_profile is None:
            return []

        # 2. Préparer le pool de candidats (exclure les vus)
        metadata = self.data_loader.load_articles_metadata()
        exclude_seen = kwargs.get("exclude_seen", True)

        if exclude_seen:
            seen = set(self.data_loader.get_user_history(user_id)[self._item_col])
            pool = metadata[~metadata["article_id"].isin(seen)]
        else:
            pool = metadata

        candidate_ids = pool["article_id"].values

        # 3. Calcul de similarité MASSIVE (NumPy)
        all_embeddings = self.data_loader.load_article_embeddings_matrix()
        candidate_matrix = all_embeddings[candidate_ids]

        # cosine_similarity peut prendre (1, 250) et (N, 250) et renvoyer (1, N)
        similarities_scores = cosine_similarity(user_profile, candidate_matrix)[0]

        # 4. Top K efficace avec argsort
        top_indices = np.argsort(similarities_scores)[::-1][:num_recommendations]

        recommendations = []
        for idx in top_indices:
            article_id = int(candidate_ids[idx])
            score = float(similarities_scores[idx])
            recommendations.append(
                self._format_recommendation(
                    article_id=article_id,
                    score=score,
                    reason=f"Basé sur vos lectures récentes (score: {score:.3f})"
                )
            )

        return recommendations
