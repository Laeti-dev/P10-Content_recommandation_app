from __future__ import annotations

from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd

import logging

from recommenders.recommender import Recommender

logger = logging.getLogger(__name__)


class ContentBasedRecommender(Recommender):
    """
    Content-based recommender.

    Uses a user profile built from clicked-article embeddings and recommends the most
    similar candidate articles by cosine similarity.
    """
    MODEL_NAME = "content-based"

    def __init__(
        self,
        *,
        data_loader,
        interactions_train_df: pd.DataFrame,
        candidate_article_ids: Optional[List[int]] = None,
        k: int = 10,
        interaction_item_col: str = "click_article_id",
        user_col: str = "user_id",
        # ---- Strategy knobs (all cosine-based) ----
        profile_strategy: Literal["mean", "recency", "last_n"] = "mean",
        last_n: int = 20,
        recency_half_life_days: float = 7.0,
        # Candidate filtering
        candidate_strategy: Literal["all", "top_categories"] = "all",
        top_categories_n: int = 3,
        # Score adjustment (re-ranking)
        score_strategy: Literal["cosine", "popularity_novelty"] = "cosine",
        popularity_penalty: float = 0.0,
        novelty_boost: float = 0.0,
        # Columns / metadata
        timestamp_col: str = "click_timestamp",
        category_col: str = "category_id",
        created_ts_col: str = "created_at_ts",
    ) -> None:
        super().__init__(
            data_loader=data_loader,
            k=k,
            interaction_item_col=interaction_item_col,
        )
        self._train_df = interactions_train_df
        self._user_col = user_col
        self._timestamp_col = timestamp_col
        self._category_col = category_col
        self._created_ts_col = created_ts_col

        self._profile_strategy = profile_strategy
        self._last_n = int(last_n)
        self._recency_half_life_days = float(recency_half_life_days)

        self._candidate_strategy = candidate_strategy
        self._top_categories_n = int(top_categories_n)

        self._score_strategy = score_strategy
        self._popularity_penalty = float(popularity_penalty)
        self._novelty_boost = float(novelty_boost)

        # Candidate pool defaults to all items present in training.
        ic = self.interaction_item_col
        if candidate_article_ids is None:
            candidate_article_ids = (
                self._train_df[ic].dropna().astype(int).unique().tolist()
            )
        self._candidate_article_ids = [int(x) for x in candidate_article_ids]

        # Preload candidate embeddings once for fast recommend calls.
        self._candidate_embeddings = self.get_item_embeddings(self._candidate_article_ids)
        if self._candidate_embeddings.ndim != 2:
            raise ValueError(
                "expected candidate embeddings matrix of shape (n_items, dim)"
            )
        self._candidate_norms = np.linalg.norm(self._candidate_embeddings, axis=1)

        # Precompute candidate metadata aligned with candidate ids (for fast filtering/re-ranking).
        self._candidate_category_ids, self._candidate_created_days = self._load_candidate_metadata()
        self._candidate_popularity = self._compute_candidate_popularity()

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def _train_rows_for_user(self, user_id: int) -> pd.DataFrame:
        return self._train_df.loc[self._train_df[self._user_col] == user_id].copy()

    def _unique_recent_article_ids(self, user_rows: pd.DataFrame) -> List[int]:
        """Most-recent-first unique article ids from user rows."""
        ic = self.interaction_item_col
        if ic not in user_rows.columns:
            return []
        if self._timestamp_col in user_rows.columns:
            user_rows = user_rows.sort_values(self._timestamp_col, ascending=False)
        ordered: List[int] = []
        seen: set[int] = set()
        for v in user_rows[ic].tolist():
            if pd.isna(v):
                continue
            aid = int(v)
            if aid in seen:
                continue
            seen.add(aid)
            ordered.append(aid)
        return ordered

    def _build_user_profile_vector(self, user_id: int) -> np.ndarray:
        """
        User profile according to the selected strategy:
        - mean: simple mean of all train interactions (via base helper)
        - last_n: mean of last N unique clicked articles
        - recency: exponential time-decay weighting by click recency
        """
        ic = self.interaction_item_col
        user_rows = self._train_rows_for_user(user_id)
        if user_rows.empty:
            raise ValueError(f"no interactions found for user_id={user_id}")

        if self._profile_strategy == "mean":
            return self.build_user_profile(
                user_id,
                self._train_df,
                user_col=self._user_col,
                item_col=ic,
                weight_col="interaction_weight",
            )

        ids = self._unique_recent_article_ids(user_rows)
        if not ids:
            raise ValueError(f"no valid article ids for user_id={user_id}")

        if self._profile_strategy == "last_n":
            ids = ids[: max(1, self._last_n)]
            X = self.get_item_embeddings(ids)
            return X.mean(axis=0)

        if self._profile_strategy == "recency":
            # Build a weighted profile from (possibly duplicated) rows with exponential time decay.
            if self._timestamp_col not in user_rows.columns:
                # Fallback: no timestamps -> behave like mean.
                X = self.get_item_embeddings(ids)
                return X.mean(axis=0)

            ts = pd.to_datetime(user_rows[self._timestamp_col], unit="ms", errors="coerce")
            # If timestamps are already datetime-like, unit conversion yields NaT, so retry plain.
            if ts.isna().all():
                ts = pd.to_datetime(user_rows[self._timestamp_col], errors="coerce")
            if ts.isna().all():
                X = self.get_item_embeddings(ids)
                return X.mean(axis=0)

            max_ts = ts.max()
            age_days = (max_ts - ts).dt.total_seconds() / (3600 * 24)
            half_life = max(self._recency_half_life_days, 1e-6)
            # weight = 0.5 ** (age / half_life)
            w = np.power(0.5, age_days.to_numpy(dtype=np.float64) / half_life)

            # Align embeddings with each interaction row (not unique ids).
            row_ids = user_rows[ic].dropna().astype(int).tolist()
            X = self.get_item_embeddings(row_ids)
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                return X.mean(axis=0)
            return (X * w[:, None]).sum(axis=0) / w_sum

        raise ValueError(f"unknown profile_strategy: {self._profile_strategy}")

    def _load_candidate_metadata(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays aligned to candidate ids:
        - category_id (or -1 if missing)
        - created_age_days (or NaN if missing)
        """
        if self.data_loader is None:
            return (
                np.full(len(self._candidate_article_ids), -1, dtype=np.int64),
                np.full(len(self._candidate_article_ids), np.nan, dtype=np.float64),
            )
        meta = self.data_loader.load_articles_metadata()
        if len(meta) == 0 or "article_id" not in meta.columns:
            return (
                np.full(len(self._candidate_article_ids), -1, dtype=np.int64),
                np.full(len(self._candidate_article_ids), np.nan, dtype=np.float64),
            )

        m = meta.copy()
        if "created_date" not in m.columns and self._created_ts_col in m.columns:
            m["created_date"] = pd.to_datetime(m[self._created_ts_col], unit="ms", errors="coerce")
        if "created_date" in m.columns:
            created = pd.to_datetime(m["created_date"], errors="coerce")
            ref = pd.Timestamp.now(tz=created.dt.tz) if hasattr(created.dt, "tz") else pd.Timestamp.now()
            age_days = (ref - created).dt.total_seconds() / (3600 * 24)
            m["_created_age_days"] = age_days
        else:
            m["_created_age_days"] = np.nan

        cat_map: Dict[int, int] = {}
        if self._category_col in m.columns:
            cat_map = {
                int(a): int(c)
                for a, c in zip(m["article_id"].astype(int), m[self._category_col].fillna(-1).astype(int))
            }
        age_map: Dict[int, float] = {
            int(a): float(d) if pd.notna(d) else float("nan")
            for a, d in zip(m["article_id"].astype(int), m["_created_age_days"])
        }

        cats = np.asarray([cat_map.get(int(a), -1) for a in self._candidate_article_ids], dtype=np.int64)
        ages = np.asarray([age_map.get(int(a), float("nan")) for a in self._candidate_article_ids], dtype=np.float64)
        return cats, ages

    def _compute_candidate_popularity(self) -> np.ndarray:
        """Popularity proxy from train click counts (normalized 0..1)."""
        ic = self.interaction_item_col
        if ic not in self._train_df.columns or len(self._train_df) == 0:
            return np.zeros(len(self._candidate_article_ids), dtype=np.float64)

        counts = (
            self._train_df[ic]
            .dropna()
            .astype(int)
            .value_counts()
        )
        # Map missing ids to 0.
        raw = np.asarray([float(counts.get(int(a), 0.0)) for a in self._candidate_article_ids], dtype=np.float64)
        if raw.max() <= 0:
            return np.zeros_like(raw)
        return raw / raw.max()

    def _cosine_scores_to_candidates(self, user_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity between user_vector and every candidate embedding."""
        u = np.asarray(user_vector, dtype=np.float64).ravel()
        u_norm = float(np.linalg.norm(u))
        if u_norm == 0.0:
            # Degenerate user profile -> no signal; return zeros.
            return np.zeros(len(self._candidate_article_ids), dtype=np.float64)

        # Avoid divide-by-zero for rare zero vectors in candidate embeddings.
        denom = self._candidate_norms * u_norm
        denom = np.where(denom == 0.0, 1e-12, denom)
        return (self._candidate_embeddings @ u) / denom

    def _candidate_mask_for_user(self, user_id: int) -> np.ndarray:
        """Boolean mask over candidates based on candidate_strategy."""
        if self._candidate_strategy == "all":
            return np.ones(len(self._candidate_article_ids), dtype=bool)

        if self._candidate_strategy == "top_categories":
            user_rows = self._train_rows_for_user(user_id)
            if user_rows.empty:
                return np.zeros(len(self._candidate_article_ids), dtype=bool)

            # Build user category histogram using metadata mapping.
            cats = self._candidate_category_ids
            if cats.size == 0:
                return np.ones(len(self._candidate_article_ids), dtype=bool)

            # Map user clicked ids to categories (using candidate arrays when possible).
            clicked_ids = self._unique_recent_article_ids(user_rows)
            if not clicked_ids:
                return np.zeros(len(self._candidate_article_ids), dtype=bool)

            id_to_cat: Dict[int, int] = {
                int(a): int(c) for a, c in zip(self._candidate_article_ids, self._candidate_category_ids)
            }
            user_cats = [id_to_cat.get(int(a), -1) for a in clicked_ids]
            user_cats = [c for c in user_cats if c != -1]
            if not user_cats:
                return np.ones(len(self._candidate_article_ids), dtype=bool)

            vc = pd.Series(user_cats, dtype=np.int64).value_counts()
            top = set(vc.head(max(1, self._top_categories_n)).index.astype(int).tolist())
            return np.isin(self._candidate_category_ids, np.fromiter(top, dtype=np.int64))

        raise ValueError(f"unknown candidate_strategy: {self._candidate_strategy}")

    def _adjust_scores(self, cosine_scores: np.ndarray) -> np.ndarray:
        """Optional score adjustment: popularity penalty and novelty boost."""
        if self._score_strategy == "cosine":
            return cosine_scores

        if self._score_strategy == "popularity_novelty":
            s = cosine_scores.astype(np.float64, copy=True)

            if self._popularity_penalty != 0.0:
                s = s - self._popularity_penalty * self._candidate_popularity

            if self._novelty_boost != 0.0:
                age = self._candidate_created_days
                # Novelty is higher for smaller age. If missing, treat as neutral (0).
                if np.isfinite(age).any():
                    finite = age[np.isfinite(age)]
                    max_age = float(np.max(finite)) if finite.size else 0.0
                    denom = max(max_age, 1e-12)
                    novelty = 1.0 - np.clip(age / denom, 0.0, 1.0)
                    novelty = np.where(np.isfinite(novelty), novelty, 0.0)
                    s = s + self._novelty_boost * novelty
            return s

        raise ValueError(f"unknown score_strategy: {self._score_strategy}")

    def recommend(
        self, user_id: int, num_recommendations: int = 5, **kwargs
    ) -> List[Dict]:
        """Recommend articles to a given user (excluding already seen items)."""
        ic = self.interaction_item_col

        user_profile = self._build_user_profile_vector(user_id)

        cosine_scores = self._cosine_scores_to_candidates(user_profile)
        scores = self._adjust_scores(cosine_scores)
        mask = self._candidate_mask_for_user(user_id)
        # Mask out disallowed candidates.
        scores = np.where(mask, scores, -np.inf)

        seen = set(
            self._train_df.loc[self._train_df[self._user_col] == user_id, ic]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        # Rank candidates by score (descending), skipping seen items.
        order = np.argsort(scores)[::-1]
        recs: List[Dict] = []
        for idx in order:
            aid = int(self._candidate_article_ids[int(idx)])
            if aid in seen:
                continue
            if not np.isfinite(scores[int(idx)]):
                continue
            recs.append(
                self._format_recommendation(
                    article_id=aid,
                    score=float(scores[int(idx)]),
                    reason=(
                        f"content/{self._profile_strategy}"
                        f"+{self._candidate_strategy}"
                        f"+{self._score_strategy}"
                    ),
                )
            )
            if len(recs) >= int(num_recommendations):
                break

        return recs


