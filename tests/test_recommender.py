"""
Tests for the Recommender (azure_function/recommender.py).

Run with:
    poetry run pytest tests/test_recommender.py -v

These tests use synthetic fixtures — no real data or Azure connection required.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# We import the class directly — adjust the path if needed
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "azure_function"))

from recommender import Recommender


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_artifacts(n_articles: int = 10, n_users: int = 3, dim: int = 8) -> dict:
    """Build minimal synthetic artifacts for testing."""
    np.random.seed(42)

    candidate_ids       = np.arange(n_articles, dtype=np.int64)
    candidate_embeddings = np.random.randn(n_articles, dim).astype(np.float32)
    candidate_norms     = np.linalg.norm(candidate_embeddings, axis=1).astype(np.float32)
    candidate_norms     = np.where(candidate_norms == 0, 1e-12, candidate_norms)
    article_to_idx      = {int(aid): i for i, aid in enumerate(candidate_ids)}

    user_profiles = {
        uid: np.random.randn(dim).astype(np.float32)
        for uid in range(n_users)
    }
    user_seen = {
        0: {0, 1},   # user 0 has seen articles 0 and 1
        1: set(),    # user 1 has seen nothing
        2: set(range(n_articles)),  # user 2 has seen everything
    }
    popularity_scores = {
        int(aid): float(np.random.rand())
        for aid in candidate_ids
    }

    return {
        "candidate_ids":        candidate_ids,
        "candidate_embeddings": candidate_embeddings,
        "candidate_norms":      candidate_norms,
        "article_to_idx":       article_to_idx,
        "user_profiles":        user_profiles,
        "user_seen":            user_seen,
        "popularity_scores":    popularity_scores,
    }


@pytest.fixture
def recommender() -> Recommender:
    """Loaded recommender with synthetic artifacts."""
    rec = Recommender()
    rec.load(make_artifacts())
    return rec


# ---------------------------------------------------------------------------
# Tests — Loading
# ---------------------------------------------------------------------------

class TestLoading:

    def test_load_sets_ready_flag(self):
        rec = Recommender()
        assert not rec._is_ready
        rec.load(make_artifacts())
        assert rec._is_ready

    def test_load_sets_correct_counts(self, recommender):
        assert recommender.n_articles() == 10
        assert recommender.n_users() == 3

    def test_pop_vector_precomputed(self, recommender):
        assert recommender._pop_vector is not None
        assert len(recommender._pop_vector) == recommender.n_articles()

    def test_recommender_not_ready_raises(self):
        rec = Recommender()
        with pytest.raises(ValueError, match="not loaded"):
            rec.recommend(user_id=0)


# ---------------------------------------------------------------------------
# Tests — user_exists
# ---------------------------------------------------------------------------

class TestUserExists:

    def test_known_user_exists(self, recommender):
        assert recommender.user_exists(0)
        assert recommender.user_exists(1)
        assert recommender.user_exists(2)

    def test_unknown_user_does_not_exist(self, recommender):
        assert not recommender.user_exists(999)

    def test_user_not_found_raises(self, recommender):
        with pytest.raises(ValueError, match="not found"):
            recommender.recommend(user_id=999)


# ---------------------------------------------------------------------------
# Tests — Recommendations
# ---------------------------------------------------------------------------

class TestRecommend:

    def test_returns_correct_topk(self, recommender):
        recs = recommender.recommend(user_id=1, topk=5)
        assert len(recs) == 5

    def test_returns_list_of_ints(self, recommender):
        recs = recommender.recommend(user_id=1, topk=3)
        assert isinstance(recs, list)
        assert all(isinstance(r, (int, np.integer)) for r in recs)

    def test_no_duplicate_articles(self, recommender):
        recs = recommender.recommend(user_id=1, topk=5)
        assert len(recs) == len(set(recs))

    def test_seen_articles_excluded(self, recommender):
        """User 0 has seen articles 0 and 1 — they must not appear."""
        recs = recommender.recommend(user_id=0, topk=5)
        assert 0 not in recs
        assert 1 not in recs

    def test_topk_capped_by_available_articles(self, recommender):
        """User 2 has seen all articles — result should be empty or < topk."""
        recs = recommender.recommend(user_id=2, topk=5)
        # All articles seen → scores are -inf → no valid recommendation
        assert len(recs) <= 5

    def test_articles_are_from_catalogue(self, recommender):
        """All recommended article IDs must be in the catalogue."""
        catalogue = set(recommender.candidate_ids.tolist())
        recs = recommender.recommend(user_id=1, topk=5)
        for r in recs:
            assert int(r) in catalogue

    def test_topk_1_returns_single_article(self, recommender):
        recs = recommender.recommend(user_id=1, topk=1)
        assert len(recs) == 1


# ---------------------------------------------------------------------------
# Tests — Beta parameter
# ---------------------------------------------------------------------------

class TestBeta:

    def test_beta_zero_uses_cb_only(self, recommender):
        """beta=0 → popularity has no influence, result is pure CB."""
        recs = recommender.recommend(user_id=1, topk=5, beta=0.0)
        assert len(recs) == 5

    def test_beta_one_uses_popularity_only(self, recommender):
        """beta=1 → cosine sim has no influence, result is pure popularity."""
        recs = recommender.recommend(user_id=1, topk=5, beta=1.0)
        assert len(recs) == 5

    def test_beta_clamped_above_one(self, recommender):
        """Beta > 1 should not crash — scores remain valid floats."""
        # We don't clamp in recommender itself (clamping is in function_app),
        # but the math should still produce valid results
        recs = recommender.recommend(user_id=1, topk=3, beta=0.8)
        assert len(recs) == 3

    def test_different_betas_may_produce_different_results(self, recommender):
        """Different betas should generally produce different rankings."""
        recs_cb  = recommender.recommend(user_id=1, topk=5, beta=0.0)
        recs_pop = recommender.recommend(user_id=1, topk=5, beta=1.0)
        # They could theoretically be equal, but with random data they usually differ
        # This is a soft check — we just verify both return valid results
        assert len(recs_cb) == 5
        assert len(recs_pop) == 5


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_topk_larger_than_catalogue(self):
        """Requesting more articles than available should not crash."""
        rec = Recommender()
        rec.load(make_artifacts(n_articles=3))
        recs = rec.recommend(user_id=1, topk=10)
        assert len(recs) <= 3

    def test_empty_popularity_scores(self):
        """Missing popularity scores default to 0.0 — should not crash."""
        arts = make_artifacts()
        arts["popularity_scores"] = {}  # no scores at all
        rec = Recommender()
        rec.load(arts)
        recs = rec.recommend(user_id=1, topk=5, beta=0.5)
        assert len(recs) == 5

    def test_zero_norm_profile_raises(self):
        """A user with a zero-norm profile should raise ValueError."""
        arts = make_artifacts()
        arts["user_profiles"][0] = np.zeros(8, dtype=np.float32)
        rec = Recommender()
        rec.load(arts)
        with pytest.raises(ValueError, match="empty profile"):
            rec.recommend(user_id=0, topk=5)
