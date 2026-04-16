"""Load article embeddings and interaction tables for recommenders."""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from constants import DATA_PATH

logger = logging.getLogger(__name__)


class DataLoader:
    """ Data manager for the recommender system."""

    def __init__(self):
        self.data_path = Path(DATA_PATH)
        self._articles_metadata = None
        self._articles_embeddings = None
        self._user_interactions = None
        self._reference_date = None


    def load_articles_metadata(self) -> pd.DataFrame:
        """Load the articles metadata."""
        if self._articles_metadata is None:
            self._articles_metadata = pd.read_csv(self.data_path / "articles_metadata.csv")
        return self._articles_metadata

    def load_article_embeddings_matrix(self) -> np.ndarray:
        """
        Load the article embedding matrix (one row per article).

        The project stores this as a pickled ``numpy.ndarray`` of shape (n_articles, dim).
        Row index ``i`` matches ``article_id == i`` when metadata ids are contiguous from 0.
        """
        if self._articles_embeddings is None:
            embeddings_path = self.data_path / "articles_embeddings.pickle"
            with open(embeddings_path, "rb") as f:
                self._articles_embeddings = pickle.load(f)
        return self._articles_embeddings

    def get_embeddings_by_ids(self, article_ids: List[int]) -> np.ndarray:
        """Extract an embeddings matrix from a list of IDs (Vectorized)."""
        matrix = self.load_article_embeddings_matrix()
        # We ensure that the IDs are within the bounds to avoid an IndexError
        valid_ids = [aid for aid in article_ids if 0 <= aid < len(matrix)]
        return matrix[valid_ids]

    def load_user_interactions(self) -> pd.DataFrame:
        """Load and cache all user–article clicks from ``data/clicks/*.csv``."""
        if self._user_interactions is None:
            interactions_path = self.data_path / "clicks"
            clicks_files = sorted(interactions_path.glob("*.csv"))

            clicks_df = []
            for file_path in clicks_files:
                try:
                    df = pd.read_csv(file_path)
                    clicks_df.append(df)
                except Exception as e:
                    logger.error(f"❌ Error loading {file_path}: {e}")
            if not clicks_df:
                self._user_interactions = pd.DataFrame()
            else:
                all_interactions = pd.concat(clicks_df, ignore_index=True)
                all_interactions["click_datetime"] = pd.to_datetime(
                    all_interactions["click_timestamp"], unit="ms"
                )
                self._user_interactions = all_interactions
        return self._user_interactions

    def load_interactions_table(self) -> pd.DataFrame:
        """Alias for :meth:`load_user_interactions` (backward compatibility)."""
        return self.load_user_interactions()

    def get_user_history(self, user_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Get the history of the user."""
        interactions = self.load_user_interactions()
        user_data = interactions[interactions['user_id'] == user_id].copy()
        user_data = user_data.sort_values('click_timestamp', ascending=False)

        if limit:
            user_data = user_data.head(limit)

        return user_data

    def get_recent_popular_articles(self, days: int = None) -> pd.DataFrame:
        """Get the recent popular articles with age normalization."""
        interactions = self.load_user_interactions()
        metadata = self.load_articles_metadata()
        reference_date = self._get_reference_date()

        if len(interactions) == 0:
            logger.warning("No interactions available")
            return pd.DataFrame()

        # Aggregate all clicks by article
        popularity = interactions.groupby('click_article_id').agg({
            'user_id': 'nunique',
            'click_timestamp': 'count'
        }).rename(columns={
            'user_id': 'unique_users',
            'click_timestamp': 'total_clicks'
        })

        # Join with metadata to get the creation date
        popularity = popularity.merge(
            metadata[['article_id', 'created_date']],
            left_index=True,
            right_on='article_id',
            how='left'
        )

        # Calculate the age of the article in months
        popularity['article_age_days'] = (reference_date - popularity['created_date']).dt.days
        popularity['article_age_months'] = popularity['article_age_days'] / 30.0

        # Normalize by age (minimum 0.5 months to avoid division by zero on very recent articles)
        popularity['article_age_months'] = popularity['article_age_months'].clip(lower=0.5)

        # Raw score combining unique users and total clicks
        popularity['raw_score'] = (
            0.7 * popularity['unique_users'] + 0.3 * popularity['total_clicks']
        )

        # Score normalized by the age of the article (clicks per month of existence)
        popularity['popularity_score'] = popularity['raw_score'] / popularity['article_age_months']

        # # Temporary boost for novelty (cold start for new articles)
        # # Apply a bonus for very recent articles
        # popularity['article_age_hours'] = popularity['article_age_days'] * 24

        result = popularity.sort_values('popularity_score', ascending=False)
        logger.info(f"📈 Popular articles calculated: {len(result):,} articles")
        return result

    def get_all_users(self) -> List[int]:
        """Get the list of all users."""
        interactions = self.load_user_interactions()
        return sorted(interactions['user_id'].unique().tolist())

    def get_most_active_users(self, limit: int = 20) -> List[Dict]:
        """
        Get the most active users.

        Args:
            limit: Number of users to return

        Returns:
            List of dictionaries with user_id and number of clicks
        """
        interactions = self.load_user_interactions()

        if len(interactions) == 0:
            logger.warning("No interactions available")
            return []

        # Count clicks by user
        user_clicks = interactions.groupby('user_id').size().reset_index(name='total_clicks')
        # Count unique articles by user
        unique_articles = interactions.groupby('user_id')['click_article_id'].nunique().reset_index(name='unique_articles')
        # Merge the two
        user_stats = user_clicks.merge(unique_articles, on='user_id')
        # Sort by number of clicks descending
        user_stats = user_stats.sort_values('total_clicks', ascending=False)
        # Limit to the N users
        top_users = user_stats.head(limit)
        # Convert to list of dictionaries
        result = []
        for _, row in top_users.iterrows():
            result.append({
                'user_id': int(row['user_id']),
                'total_clicks': int(row['total_clicks']),
                'unique_articles': int(row['unique_articles'])
            })

        return result

    def get_article_info(self, article_id: int) -> Dict:
        """Get the information of an article."""
        metadata = self.load_articles_metadata()
        article = metadata[metadata['article_id'] == article_id]

        if len(article) == 0:
            return {"error": "Article not found"}

        article_dict = article.iloc[0].to_dict()

        # Convert numpy types to JSON serializable types
        for key, value in article_dict.items():
            if hasattr(value, 'item'):  # numpy types
                article_dict[key] = value.item()
            elif pd.isna(value):
                article_dict[key] = None

        return article_dict

data_loader = DataLoader()
