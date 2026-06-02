"""
FastAPI app — Recommendation System Management Interface.
Displays a list of users, calls Azure Function, and shows top-5 recommendations.

Run locally:
    uvicorn app.main:app --reload --port 8000

Environment variables:
    AZURE_FUNCTION_URL : Base URL of the Azure Function (default: http://localhost:7071/api)
"""

from __future__ import annotations

import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Recommendation System — Management App",
    description="Interface for testing and managing the CBRecommender.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

FUNCTION_URL = os.environ.get("AZURE_FUNCTION_URL", "http://localhost:7071/api")

# Static user list for the demo — replace with a DB call in production
DEMO_USERS: List[int] = [
    11, 27, 42, 65, 88,
    103, 157, 201, 334, 512,
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page — displays the list of demo users."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "users": DEMO_USERS,
        "function_url": FUNCTION_URL,
    })


@app.get("/recommend/{user_id}")
async def recommend(user_id: int, beta: float = 0.8, topk: int = 5):
    """
    Call the Azure Function and return recommendations for a given user.

    Args:
        user_id : User ID to get recommendations for.
        beta    : Weight of popularity vs semantic score (0=CB only, 1=popularity only).
        topk    : Number of recommendations to return.

    Returns:
        JSON with user_id and list of recommended article IDs.
    """
    url = f"{FUNCTION_URL}/recommend/{user_id}"
    params = {"beta": beta, "topk": topk}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Azure Function timed out for user {user_id}."
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Azure Function error: {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Azure Function at {FUNCTION_URL}. Is it running?"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "function_url": FUNCTION_URL}
