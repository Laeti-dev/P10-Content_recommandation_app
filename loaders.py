"""Load article embeddings and interaction tables for recommenders."""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from constants import DATA_PATH

logger = logging.getLogger(__name__)


def _interaction_event_times(df: pd.DataFrame) -> pd.Series:
    if "click_datetime" in df.columns:
        return pd.to_datetime(df["click_datetime"], errors="coerce")
    ts = df["click_timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        return pd.to_datetime(ts, unit="ms", errors="coerce")
    return pd.to_datetime(ts, errors="coerce")


def subsample_validation_users(
    df_val: pd.DataFrame,
    *,
    max_users: Optional[int] = None,
    seed: Optional[int] = None,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Return a validation frame limited to at most ``max_users`` distinct users."""
    if max_users is None:
        return df_val.copy()
    unique_users = df_val[user_col].dropna().unique()
    if len(unique_users) <= max_users:
        return df_val.copy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique_users, size=int(max_users), replace=False)
    return df_val.loc[df_val[user_col].isin(chosen)].copy()


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
        matrix = self.load_article_embeddings_matrix()
        # matrix dimensions
        logging.info(f"Matrix dimensions: {matrix.shape}")
        # Check that the IDs are valid
        valid_ids = [int(aid) for aid in article_ids if 0 <= aid < len(matrix)]
        logging.info(f"Valid IDs: {len(valid_ids)}")
        return matrix[valid_ids]

    def embeddings_aligned_to_article_ids(
        self,
        article_ids: Iterable[int],
        *,
        drop_out_of_range: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sorted unique article ids and embedding rows in the same order (row i ↔ ``ids[i]``).

        Pickle matrix rows are indexed by ``article_id`` for this dataset (ids 0 .. n_rows-1).

        Args:
            article_ids: e.g. union of ``article_id`` values from train and test clicks.
            drop_out_of_range: If True, omit ids outside ``[0, n_rows)``. If False, raise.
        """
        matrix = self.load_article_embeddings_matrix()
        raw = np.unique(np.asarray(list(article_ids), dtype=np.int64))
        n = len(matrix)
        ok = (raw >= 0) & (raw < n)
        if not np.all(ok):
            bad = raw[~ok]
            if drop_out_of_range:
                logger.warning(
                    "embeddings_aligned_to_article_ids: dropping %d article_ids "
                    "outside [0, %d) (examples: %s)",
                    int(bad.size),
                    n,
                    bad[: min(5, bad.size)].tolist(),
                )
                raw = raw[ok]
            else:
                raise ValueError(
                    f"article_ids out of embedding range [0, {n}): {bad[:20].tolist()}"
                )
        if raw.size == 0:
            raise ValueError("no valid article_ids for embedding lookup")
        return raw, matrix[raw]

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
        """Load the articles metadata"""
        if self._articles_metadata is None:
            metadata_path = self.data_path / "articles_metadata.csv"
            df = pd.read_csv(metadata_path)

            # Add the creation date
            df['created_date'] = pd.to_datetime(df['created_at_ts'], unit='ms')

            self._articles_metadata = df
            logger.info(f"Articles metadata loaded: {len(df):,} articles")

        return self._articles_metadata



    def get_user_history(self, user_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Get the history of the user."""
        interactions = self.load_user_interactions()
        user_data = interactions[interactions['user_id'] == user_id].copy()
        user_data = user_data.sort_values('click_timestamp', ascending=False)

        if limit:
            user_data = user_data.head(limit)

        return user_data

    def _get_reference_date(self) -> pd.Timestamp:
        """Latest click time in loaded interactions, else latest article ``created_date``."""
        if self._reference_date is not None:
            return self._reference_date
        interactions = self.load_user_interactions()
        try:
            if len(interactions) > 0 :
                ref = interactions["click_datetime"].max()
            else:
                metadata = self.load_articles_metadata()
                ref = pd.to_datetime(metadata["created_date"]).max()
        except Exception as e:
            logger.error(f"Error getting reference date: {e}")
            ref = pd.Timestamp.now()
        self._reference_date = pd.Timestamp(ref)
        return self._reference_date

    def data_split(
        self,
        interactions: pd.DataFrame,
        *,
        min_clicks_per_user: int = 5,
        keep_duplicate_clicks: bool = False,
        duplicate_weight_mode: Optional[Literal["count"]] = None,
        item_col: str = "click_article_id",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a temporal train/validation split on click interactions.

        Only users with at least ``min_clicks_per_user`` interactions (rows, after
        optional deduplication) are kept. For each such user, the chronologically
        last click is held out as validation; all earlier clicks stay in train.
        That validation article is unseen in training for that user.

        Args:
            interactions: Must include ``user_id``, ``item_col``, and
                ``click_timestamp`` (numeric ms or datetime-like).
            min_clicks_per_user: Minimum interaction count per user after
                duplicate handling.
            keep_duplicate_clicks: If False, collapse repeated (user, article)
                clicks to a single row, keeping the latest ``click_timestamp``.
                If True, keep every click row.
            duplicate_weight_mode: Only used when ``keep_duplicate_clicks`` is
                True. If ``"count"``, adds ``interaction_weight`` = number of
                clicks for that (user, article) pair (same value on each duplicate
                row). If None, no weight column is added.
            item_col: Column name for the article id (e.g. ``click_article_id``
                from raw clicks or ``article_id`` from merged EDA exports).

        Returns:
            (train_df, val_df) with the same columns as the prepared interactions
            (plus ``interaction_weight`` when requested).

        Raises:
            ValueError: If required columns are missing or ``duplicate_weight_mode``
                is set while ``keep_duplicate_clicks`` is False.
        """
        required = {"user_id", "click_timestamp", item_col}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing columns: {sorted(missing)}")

        if duplicate_weight_mode is not None and not keep_duplicate_clicks:
            raise ValueError(
                "duplicate_weight_mode is only supported when keep_duplicate_clicks=True"
            )
        if duplicate_weight_mode not in (None, "count"):
            raise ValueError('duplicate_weight_mode must be None or "count"')

        df = interactions.loc[:, list(interactions.columns)].copy()
        sort_keys: List[str] = ["user_id", "click_timestamp"]
        if "click_datetime" in df.columns:
            sort_keys.append("click_datetime")
        # Stable ordering when timestamps tie
        df["_row_order"] = np.arange(len(df), dtype=np.int64)
        sort_keys.append("_row_order")
        df = df.sort_values(sort_keys, kind="mergesort")

        pair_keys = ["user_id", item_col]
        if keep_duplicate_clicks:
            if duplicate_weight_mode == "count":
                df["interaction_weight"] = df.groupby(pair_keys, sort=False)[
                    "click_timestamp"
                ].transform("size")
        else:
            df = df.drop_duplicates(subset=pair_keys, keep="last")

        df = df.drop(columns=["_row_order"])

        user_counts = df.groupby("user_id", sort=False).size()
        eligible_users = user_counts[user_counts >= min_clicks_per_user].index
        df = df[df["user_id"].isin(eligible_users)]

        val_idx = df.groupby("user_id", sort=False).tail(1).index
        val_df = df.loc[val_idx].copy()
        train_df = df.drop(val_idx).copy()

        print(f'Total number of users in train: {train_df["user_id"].nunique()}')
        print(f'Total number of users in val: {val_df["user_id"].nunique()}')

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def data_split_by_date(
        self,
        interactions: pd.DataFrame,
        *,
        split_time: pd.Timestamp,
        val_end_exclusive: Optional[pd.Timestamp] = None,
        min_clicks_per_user_train: int = 1,
        min_clicks_per_user_val: int = 1,
        item_col: str = "article_id",
        split_granularity: Literal["day", "instant"] = "day",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions by calendar time: train before ``split_time``, val on/after.

        ``split_granularity`` is ``"day"`` (compare normalized dates) or ``"instant"``
        (compare full timestamps). Users must meet minimum click counts in both splits.
        """
        if val_end_exclusive is not None and split_time >= val_end_exclusive:
            raise ValueError(
                "split_time must be strictly before val_end_exclusive when both are set"
            )

        required = {"user_id", "click_timestamp", item_col}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing columns: {sorted(missing)}")

        if split_granularity not in ("day", "instant"):
            raise ValueError('split_granularity must be "day" or "instant"')

        df = interactions.loc[:, list(interactions.columns)].copy()
        event_times = _interaction_event_times(df)
        split_ts = pd.Timestamp(split_time)

        if split_granularity == "day":
            boundary = split_ts.normalize()
            train_mask = event_times.dt.normalize() < boundary
            val_mask = event_times.dt.normalize() >= boundary
        else:
            train_mask = event_times < split_ts
            val_mask = event_times >= split_ts

        if val_end_exclusive is not None:
            val_end = pd.Timestamp(val_end_exclusive)
            val_mask = val_mask & (event_times < val_end)

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()

        train_counts = train_df.groupby("user_id", sort=False).size()
        val_counts = val_df.groupby("user_id", sort=False).size()
        eligible_train = set(
            train_counts[train_counts >= min_clicks_per_user_train].index
        )
        eligible_val = set(val_counts[val_counts >= min_clicks_per_user_val].index)
        eligible_users = eligible_train & eligible_val

        train_df = train_df[train_df["user_id"].isin(eligible_users)].reset_index(
            drop=True
        )
        val_df = val_df[val_df["user_id"].isin(eligible_users)].reset_index(drop=True)
        return train_df, val_df

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
            metadata[['article_id', 'created_at_ts']],
            left_index=True,
            right_on='article_id',
            how='left'
        )

        # Calculate the age of the article in months
        popularity['created_date'] = pd.to_datetime(popularity['created_at_ts'], unit='ms')
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

    def get_category_map(
        self, article_ids: Optional[Iterable[int]] = None
    ) -> Dict[int, int]:
        """Return ``article_id → category_id`` from cached article metadata."""
        metadata = self.load_articles_metadata()
        full = dict(
            zip(
                metadata["article_id"].astype(int),
                metadata["category_id"].astype(int),
            )
        )
        if article_ids is None:
            return full
        return {int(aid): full[int(aid)] for aid in article_ids if int(aid) in full}

    def get_article_created_ts_map(
        self, article_ids: Optional[Iterable[int]] = None
    ) -> Dict[int, int]:
        """Return ``article_id → created_at_ts`` (ms) from cached article metadata."""
        metadata = self.load_articles_metadata()
        full = dict(
            zip(
                metadata["article_id"].astype(int),
                metadata["created_at_ts"].astype(np.int64),
            )
        )
        if article_ids is None:
            return full
        return {int(aid): int(full[int(aid)]) for aid in article_ids if int(aid) in full}

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
