import logging
from enum import Enum

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi_cache.decorator import cache
from pydantic import Field

from app.utils import HfBaseModel, handle_exception
from recommender.main import get_tgi_config
from recommender.utils.calcuation import TGIConfig
from recommender.utils.const import (
    AWS_INFERENCE_INSTANCE_TYPES,
    GOOGLE_CLOUD_INFERENCE_INSTANCE_TYPES,
    HUGGINGFACE_INSTANCE_TYPES,
)

router = APIRouter()

logger = logging.getLogger("uvicorn")

INSTANCE_TYPE_MAP = {
    "gcp": GOOGLE_CLOUD_INFERENCE_INSTANCE_TYPES["gpu"],
    "aws": AWS_INFERENCE_INSTANCE_TYPES["gpu"],
    "hf": HUGGINGFACE_INSTANCE_TYPES["gpu"],
}


class Provider(str, Enum):
    hf = "hf"
    gcp = "gcp"
    aws = "aws"


class QueryParameter(HfBaseModel):
    model_id: str = Field(None, description="Hugging Face Model ID")
    gpu_memory: int = Field(999, description="GPU Memory in GB")


class RecommendationResponse(HfBaseModel):
    model_id: str
    instance: str
    configuration: TGIConfig


@router.get("/provider/{provider}/recommend")
@cache(expire=3600 * 3)  # cache 3 hours
async def recommend(provider: Provider, params: QueryParameter = Depends()):
    if params.model_id == "":
        # Returns 200 response to be cacheable
        return JSONResponse({"error": "No modelid provided", "recommendation": []}, status_code=200)
    # filter potential instance types based on requested memory
    _instance_map = [config for config in INSTANCE_TYPE_MAP[provider] if config["memoryInGB"] <= params.gpu_memory]
    # Try to create model recommendation
    try:
        for config in _instance_map:
            tgi_config = get_tgi_config(
                model_id=params.model_id,
                gpu_memory=config["memoryInGB"],
                num_gpus=config["numGpus"],
            )
            if tgi_config:
                break  # break after first valid config to get smallest instance
    except Exception as e:
        logger.error(f"Error while getting recommendation: {e}")
        return handle_exception(e)
    # Return custom 200 response with error to be cacheable
    if not tgi_config:
        return JSONResponse({"error": "No recommendation found"}, status_code=200)

    return RecommendationResponse(
        model_id=params.model_id,
        instance=config["name"],
        configuration=tgi_config,
    )
