import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from app.main import app as fastapi_app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    app: FastAPI = fastapi_app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

