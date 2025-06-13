from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import os
import logging
import tensorflow as tf
from app.db import database
from app.db.database import Base
from app import models
from app.routers import (
    auth,
    wardrobe,
    outfits,
    weekly_plans,
    occasions,
    style_history,
    statistics,
    ai_analyzer,
    recommendations,
    user_profile,
    community,
)

# Disable oneDNN optimizations if desired
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Use FastAPI lifespan event to manage app startup and shutdown.
    Execute DDL statements in development mode only.
    """
    if os.getenv("ENV") == "development" and os.getenv("RUN_MAIN") == "true":
        Base.metadata.drop_all(bind=database.engine)
        Base.metadata.create_all(bind=database.engine)
    yield

app = FastAPI(lifespan=lifespan)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5713",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5713",
    "https://digital-wardrobe-system.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API routers
app.include_router(auth.router, prefix="/api")
app.include_router(wardrobe.router, prefix="/api")
app.include_router(outfits.router, prefix="/api")
app.include_router(weekly_plans.router, prefix="/api")
app.include_router(occasions.router, prefix="/api")
app.include_router(style_history.router, prefix="/api")
app.include_router(statistics.router, prefix="/api")
app.include_router(ai_analyzer.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(user_profile.router, prefix="/api")
app.include_router(community.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Entry point
if __name__ == "__main__":
    # In development, reload=True will set RUN_MAIN for our startup guard
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
