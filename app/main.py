import logging
import os

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from app.routes import config
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
router = APIRouter()

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from the specified origins
    allow_credentials=True,  # Allows cookies to be included in HTTP requests
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logger = logging.getLogger("uvicorn")

# Redis connection
redis = None

# IDS
# Gated: meta-llama/Llama-2-7b-chat-hf
# Private:
# Public: HuggingFaceH4/zephyr-7b-beta
# False: meta-llama/Llama-2-7b

app.include_router(config.router)


@app.on_event("startup")
async def startup():
    if not os.getenv("HF_TOKEN", None):
        logger.info("No HF_TOKEN found in environment. Cannot look up gated models.")

    if os.getenv("REDIS_URL", None):
        global redis
        logger.info(f"Using RedisBackend at {os.getenv('REDIS_URL')}")
        redis = await aioredis.from_url(os.getenv("REDIS_URL"))
        FastAPICache.init(RedisBackend(redis), prefix="optimum-cache")
    else:
        FastAPICache.init(InMemoryBackend())
        logger.info("Using InMemoryBackend for cache")