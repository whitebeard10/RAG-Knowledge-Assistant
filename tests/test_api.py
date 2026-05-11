import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

@pytest.mark.asyncio
async def test_query_endpoint_structure():
    # This test only checks the API structure, as it would fail without real API keys
    # In a real CI/CD, we would mock the RAGService
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Intentionally sending invalid data to see if Pydantic catches it
        response = await ac.post("/api/v1/query", json={"not_a_query": "test"})
    assert response.status_code == 422 # Unprocessable Entity
