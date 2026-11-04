"""API tests for the ReelRank recommendation service."""

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["health"] == "/health"


def test_health_endpoint_structure():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ranking_model_loaded" in data
    assert "faiss_index_loaded" in data
    assert "faiss_movie_count" in data
    assert "user_embedding_count" in data
    assert "redis_connected" in data
    assert data["cache_ttl_seconds"] == 300


def test_recommendation_rejects_zero_user_id():
    response = client.get("/recommend/0")
    assert response.status_code == 400
    assert response.json()["detail"] == "user_id must be positive."


def test_recommendation_rejects_negative_user_id():
    response = client.get("/recommend/-1")
    assert response.status_code == 400


def test_recommendation_rejects_limit_below_minimum():
    response = client.get("/recommend/2?limit=0")
    assert response.status_code == 422


def test_recommendation_rejects_limit_above_maximum():
    response = client.get("/recommend/2?limit=51")
    assert response.status_code == 422


def test_recommendation_rejects_candidates_below_minimum():
    response = client.get("/recommend/2?candidates=9")
    assert response.status_code == 422


def test_recommendation_rejects_candidates_above_maximum():
    response = client.get("/recommend/2?candidates=501")
    assert response.status_code == 422


def test_unknown_movie_returns_404():
    response = client.get("/movie/999999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Movie ID not found."


def test_recommendation_response_for_unknown_user():
    response = client.get("/recommend/999999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()