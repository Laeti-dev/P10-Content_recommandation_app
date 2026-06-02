"""
Azure Function — Recommendation endpoint.

Routes:
    GET /api/recommend/{user_id}?beta=0.8&topk=5
    GET /api/health

Environment variables (set in Azure Function App > Configuration):
    AZURE_STORAGE_CONNECTION_STRING : Azure Blob Storage connection string
    ARTIFACTS_CONTAINER             : Blob container name (default: artifacts)
    DEFAULT_BETA                    : Default beta value (default: 0.8)
    DEFAULT_TOPK                    : Default top-K (default: 5)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

import azure.functions as func
import numpy as np
from azure.storage.blob import BlobServiceClient

from recommender import ProductionRecommender

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure Function App
# ---------------------------------------------------------------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------------------------------------------------------------------------
# Global cache — loaded once per warm instance
# ---------------------------------------------------------------------------
_recommender: Optional[ProductionRecommender] = None


def _get_recommender() -> ProductionRecommender:
    """
    Load and cache the recommender.
    On a warm Azure Function instance, artifacts are loaded only once.
    """
    global _recommender
    if _recommender is not None:
        return _recommender

    logger.info("Cold start — loading artifacts from Azure Blob Storage...")

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise EnvironmentError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Add it in Function App > Configuration > Application settings."
        )

    container_name = os.environ.get("ARTIFACTS_CONTAINER", "recommenderv1")

    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    artifact_names = [
        "candidate_ids",
        "candidate_embeddings",
        "candidate_norms",
        "article_to_idx",
        "user_profiles",
        "user_seen",
        "popularity_scores",
    ]

    artifacts: Dict[str, Any] = {}
    for name in artifact_names:
        blob_client = container_client.get_blob_client(f"{name}.pkl")
        data = blob_client.download_blob().readall()
        artifacts[name] = pickle.loads(data)
        logger.info("  ✅ %s loaded", name)

    _recommender = ProductionRecommender().load(artifacts)
    logger.info(
        "Recommender ready — %d articles, %d users.",
        _recommender.n_articles(),
        _recommender.n_users(),
    )
    return _recommender


def _json_response(data: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(data),
        mimetype="application/json",
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route(route="recommend/{user_id}", methods=["GET"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Return top-K recommendations for a given user.

    Path params:
        user_id (int) : User ID

    Query params:
        beta  (float) : Popularity weight, default 0.8
        topk  (int)   : Number of recommendations, default 5
    """
    # --- Parse user_id ---
    try:
        user_id = int(req.route_params.get("user_id"))
    except (TypeError, ValueError):
        return _json_response(
            {"error": "user_id must be an integer."},
            status_code=400,
        )

    # --- Parse query params ---
    try:
        beta = float(req.params.get("beta", os.environ.get("DEFAULT_BETA", "0.8")))
        beta = max(0.0, min(1.0, beta))  # clamp to [0, 1]
    except ValueError:
        return _json_response({"error": "beta must be a float between 0 and 1."}, status_code=400)

    try:
        topk = int(req.params.get("topk", os.environ.get("DEFAULT_TOPK", "5")))
        topk = max(1, min(topk, 50))  # clamp to [1, 50]
    except ValueError:
        return _json_response({"error": "topk must be an integer."}, status_code=400)

    # --- Load recommender ---
    try:
        recommender = _get_recommender()
    except EnvironmentError as e:
        logger.error("Configuration error: %s", e)
        return _json_response({"error": str(e)}, status_code=500)
    except Exception as e:
        logger.error("Failed to load recommender: %s", e)
        return _json_response(
            {"error": "Failed to load recommendation model. Check logs."},
            status_code=500,
        )

    # --- Check user exists ---
    if not recommender.user_exists(user_id):
        return _json_response(
            {
                "error": f"User {user_id} not found.",
                "hint": "This user has no interaction history in the training data.",
            },
            status_code=404,
        )

    # --- Get recommendations ---
    try:
        recommendations = recommender.recommend(user_id, topk=topk, beta=beta)
        return _json_response({
            "user_id": user_id,
            "recommendations": recommendations,
            "beta": beta,
            "topk": topk,
        })
    except Exception as e:
        logger.error("Recommendation error for user %d: %s", user_id, e)
        return _json_response(
            {"error": "Recommendation failed.", "detail": str(e)},
            status_code=500,
        )


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check — also reports recommender status."""
    try:
        recommender = _get_recommender()
        return _json_response({
            "status": "ok",
            "n_articles": recommender.n_articles(),
            "n_users": recommender.n_users(),
            "model": "CBRecommender + popularity",
        })
    except Exception as e:
        return _json_response(
            {"status": "error", "detail": str(e)},
            status_code=500,
        )
