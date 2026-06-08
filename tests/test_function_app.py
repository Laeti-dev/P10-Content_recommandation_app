"""
Tests for the Azure Function endpoints (azure_function/function_app.py).

These tests mock Azure SDK and the recommender to avoid loading real artifacts.
Run with:
    poetry run pytest tests/test_function_app.py -v
"""

import json
import os
import pickle
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock Azure packages before importing function_app
# ---------------------------------------------------------------------------

def _ensure_package(name: str) -> types.ModuleType:
    """Register a fake package in sys.modules (needs __path__ for sub-imports)."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    return sys.modules[name]


_ensure_package("azure")
azure_functions = _ensure_package("azure.functions")


def _identity_decorator(*_args, **_kwargs):
    """Pass-through route decorator so endpoint functions stay callable."""
    def decorator(fn):
        return fn
    return decorator


class _FakeHttpResponse:
    def __init__(self, body, mimetype=None, status_code=200):
        self.body = body.encode() if isinstance(body, str) else body
        self.mimetype = mimetype
        self.status_code = status_code


mock_app = MagicMock()
mock_app.route = _identity_decorator
azure_functions.FunctionApp = MagicMock(return_value=mock_app)
azure_functions.HttpRequest = MagicMock
azure_functions.HttpResponse = _FakeHttpResponse
azure_functions.AuthLevel = MagicMock()
azure_functions.AuthLevel.ANONYMOUS = "ANONYMOUS"

_ensure_package("azure.storage")
azure_storage_blob = _ensure_package("azure.storage.blob")
azure_storage_blob.BlobServiceClient = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "azure_function"))

import function_app  # noqa: E402 — must load after Azure mocks
from recommender import Recommender  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_recommender_cache():
    """Isolate tests that exercise _get_recommender caching."""
    function_app._recommender = None
    yield
    function_app._recommender = None


def make_artifacts(n_articles: int = 5, n_users: int = 2, dim: int = 4) -> dict:
    """Minimal synthetic artifacts for _get_recommender integration tests."""
    candidate_ids = np.arange(n_articles, dtype=np.int64)
    candidate_embeddings = np.random.randn(n_articles, dim).astype(np.float32)
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1).astype(np.float32)
    return {
        "candidate_ids": candidate_ids,
        "candidate_embeddings": candidate_embeddings,
        "candidate_norms": candidate_norms,
        "article_to_idx": {int(aid): i for i, aid in enumerate(candidate_ids)},
        "user_profiles": {0: np.random.randn(dim).astype(np.float32)},
        "user_seen": {0: set()},
        "popularity_scores": {int(aid): 0.5 for aid in candidate_ids},
    }


def make_mock_request(user_id: str, params: dict | None = None):
    """Build a mock HttpRequest."""
    req = MagicMock()
    req.route_params = {"user_id": user_id}
    req.params = params or {}
    return req


def make_mock_recommender(recommendations=None, n_articles=364047, n_users=64734):
    """Build a mock Recommender."""
    rec = MagicMock()
    rec._is_ready = True
    rec.n_articles.return_value = n_articles
    rec.n_users.return_value = n_users
    rec.user_exists.return_value = True
    rec.recommend.return_value = recommendations or [160974, 96210, 234698, 336221, 331116]
    return rec


def parse_json_response(response) -> dict:
    """Parse the JSON body of an HttpResponse."""
    return json.loads(response.body)


def _setup_blob_mocks(artifacts: dict):
    """Wire BlobServiceClient to return pickled artifacts for each blob."""
    def get_blob_client(blob_name: str):
        client = MagicMock()
        key = blob_name.removesuffix(".pkl")
        client.download_blob.return_value.readall.return_value = pickle.dumps(artifacts[key])
        return client

    mock_container = MagicMock()
    mock_container.get_blob_client.side_effect = get_blob_client

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    return mock_service


# ---------------------------------------------------------------------------
# Tests — _json_response
# ---------------------------------------------------------------------------

class TestJsonResponse:

    def test_json_response_sets_mimetype_and_status(self):
        response = function_app._json_response({"ok": True}, status_code=201)

        assert response.status_code == 201
        assert response.mimetype == "application/json"
        assert parse_json_response(response) == {"ok": True}


# ---------------------------------------------------------------------------
# Tests — _get_recommender
# ---------------------------------------------------------------------------

class TestGetRecommender:

    def test_raises_when_connection_string_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="AZURE_STORAGE_CONNECTION_STRING"):
                function_app._get_recommender()

    def test_uses_artifacts_container_env_var(self):
        artifacts = make_artifacts()
        mock_service = _setup_blob_mocks(artifacts)

        with patch.dict(
            os.environ,
            {
                "AZURE_STORAGE_CONNECTION_STRING": "test-conn",
                "ARTIFACTS_CONTAINER": "my-container",
            },
        ):
            with patch("function_app.BlobServiceClient") as mock_bsc:
                mock_bsc.from_connection_string.return_value = mock_service
                function_app._get_recommender()

        mock_service.get_container_client.assert_called_once_with("my-container")

    def test_default_container_name_is_recommenderv1(self):
        artifacts = make_artifacts()
        mock_service = _setup_blob_mocks(artifacts)

        with patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-conn"}):
            with patch("function_app.BlobServiceClient") as mock_bsc:
                mock_bsc.from_connection_string.return_value = mock_service
                function_app._get_recommender()

        mock_service.get_container_client.assert_called_once_with("recommenderv1")

    def test_loads_all_artifact_blobs(self):
        artifacts = make_artifacts()
        mock_service = _setup_blob_mocks(artifacts)

        with patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-conn"}):
            with patch("function_app.BlobServiceClient") as mock_bsc:
                mock_bsc.from_connection_string.return_value = mock_service
                recommender = function_app._get_recommender()

        container = mock_service.get_container_client.return_value
        blob_names = [call.args[0] for call in container.get_blob_client.call_args_list]
        assert blob_names == [
            "candidate_ids.pkl",
            "candidate_embeddings.pkl",
            "candidate_norms.pkl",
            "article_to_idx.pkl",
            "user_profiles.pkl",
            "user_seen.pkl",
            "popularity_scores.pkl",
        ]
        assert isinstance(recommender, Recommender)
        assert recommender._is_ready

    def test_caches_recommender_on_second_call(self):
        artifacts = make_artifacts()
        mock_service = _setup_blob_mocks(artifacts)

        with patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-conn"}):
            with patch("function_app.BlobServiceClient") as mock_bsc:
                mock_bsc.from_connection_string.return_value = mock_service
                first = function_app._get_recommender()
                second = function_app._get_recommender()

        assert first is second
        mock_bsc.from_connection_string.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — /recommend endpoint
# ---------------------------------------------------------------------------

class TestRecommendEndpoint:

    def test_valid_request_returns_recommendations(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.return_value = make_mock_recommender()

            response = function_app.recommend(make_mock_request("5890"))
            body = parse_json_response(response)

            assert response.status_code == 200
            assert body["user_id"] == 5890
            assert body["beta"] == 0.8
            assert body["topk"] == 5
            assert len(body["recommendations"]) == 5

    def test_invalid_user_id_returns_400(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.return_value = make_mock_recommender()

            response = function_app.recommend(make_mock_request("not_an_int"))
            body = parse_json_response(response)

            assert response.status_code == 400
            assert body["error"] == "user_id must be an integer."
            mock_get.assert_not_called()

    def test_unknown_user_returns_404(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_rec.user_exists.return_value = False
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("99999"))
            body = parse_json_response(response)

            assert response.status_code == 404
            assert "User 99999 not found." in body["error"]
            assert "hint" in body

    def test_invalid_beta_returns_400(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.return_value = make_mock_recommender()

            response = function_app.recommend(make_mock_request("5890", params={"beta": "abc"}))
            body = parse_json_response(response)

            assert response.status_code == 400
            assert "beta" in body["error"]

    def test_invalid_topk_returns_400(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.return_value = make_mock_recommender()

            response = function_app.recommend(make_mock_request("5890", params={"topk": "xyz"}))
            body = parse_json_response(response)

            assert response.status_code == 400
            assert "topk" in body["error"]

    def test_beta_param_is_passed_to_recommender(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            function_app.recommend(make_mock_request("5890", params={"beta": "0.5"}))

            mock_rec.recommend.assert_called_once_with(5890, topk=5, beta=0.5)

    def test_topk_param_is_passed_to_recommender(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            function_app.recommend(make_mock_request("5890", params={"topk": "3"}))

            mock_rec.recommend.assert_called_once_with(5890, topk=3, beta=0.8)

    def test_beta_clamped_to_one(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("5890", params={"beta": "2.5"}))
            body = parse_json_response(response)

            mock_rec.recommend.assert_called_once_with(5890, topk=5, beta=1.0)
            assert body["beta"] == 1.0

    def test_beta_clamped_to_zero(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("5890", params={"beta": "-0.5"}))
            body = parse_json_response(response)

            mock_rec.recommend.assert_called_once_with(5890, topk=5, beta=0.0)
            assert body["beta"] == 0.0

    def test_topk_clamped_to_50(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("5890", params={"topk": "100"}))
            body = parse_json_response(response)

            mock_rec.recommend.assert_called_once_with(5890, topk=50, beta=0.8)
            assert body["topk"] == 50

    def test_topk_clamped_to_1(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("5890", params={"topk": "0"}))
            body = parse_json_response(response)

            mock_rec.recommend.assert_called_once_with(5890, topk=1, beta=0.8)
            assert body["topk"] == 1

    def test_default_beta_from_env(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            with patch.dict(os.environ, {"DEFAULT_BETA": "0.3"}):
                function_app.recommend(make_mock_request("5890"))

            mock_rec.recommend.assert_called_once_with(5890, topk=5, beta=0.3)

    def test_default_topk_from_env(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_get.return_value = mock_rec

            with patch.dict(os.environ, {"DEFAULT_TOPK": "7"}):
                function_app.recommend(make_mock_request("5890"))

            mock_rec.recommend.assert_called_once_with(5890, topk=7, beta=0.8)

    def test_recommender_load_environment_error_returns_500(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.side_effect = EnvironmentError("Missing config")

            response = function_app.recommend(make_mock_request("5890"))
            body = parse_json_response(response)

            assert response.status_code == 500
            assert body["error"] == "Missing config"

    def test_recommender_load_generic_error_returns_500(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.side_effect = RuntimeError("Blob unavailable")

            response = function_app.recommend(make_mock_request("5890"))
            body = parse_json_response(response)

            assert response.status_code == 500
            assert body["error"] == "Failed to load recommendation model. Check logs."

    def test_recommendation_failure_returns_500_with_detail(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_rec = make_mock_recommender()
            mock_rec.recommend.side_effect = ValueError("bad profile")
            mock_get.return_value = mock_rec

            response = function_app.recommend(make_mock_request("5890"))
            body = parse_json_response(response)

            assert response.status_code == 500
            assert body["error"] == "Recommendation failed."
            assert body["detail"] == "bad profile"


# ---------------------------------------------------------------------------
# Tests — /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_ok_when_recommender_ready(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.return_value = make_mock_recommender(n_articles=100, n_users=20)

            response = function_app.health(MagicMock())
            body = parse_json_response(response)

            assert response.status_code == 200
            assert body["status"] == "ok"
            assert body["n_articles"] == 100
            assert body["n_users"] == 20
            assert body["model"] == "CBRecommender + popularity"

    def test_health_returns_error_when_recommender_fails(self):
        with patch("function_app._get_recommender") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            response = function_app.health(MagicMock())
            body = parse_json_response(response)

            assert response.status_code == 500
            assert body["status"] == "error"
            assert body["detail"] == "Connection failed"
