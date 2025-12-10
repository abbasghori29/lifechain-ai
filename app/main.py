from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import get_settings


def get_application() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.APP_NAME, version="1.0.0")

    # CORS - Allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = get_application()


