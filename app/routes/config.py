from dataclasses import asdict
import logging
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field

from app.utils import HfBaseModel, handle_exception
from recommender.main import get_tgi_config
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from recommender.utils.calcuation import TGIConfig

router = APIRouter()

logger = logging.getLogger("uvicorn")


from fastapi_cache.decorator import cache


class PathParams(HfBaseModel):
    model_id: str = Field(None, description="Hugging Face Model ID to lookup")
    gpu_memory: int = Field(
        0,
        description="GPU memory in GB",
    )
    num_gpus: int = Field(1, description="Number of GPUs to use")


class ConfigResponse(HfBaseModel):
    config: TGIConfig


@router.get("/tgi/config")
@cache(expire=3600 * 3)  # cache 3 hours
async def config(params: PathParams = Depends()):
    print(params)
    if params.model_id == "" or params.gpu_memory == 0:
        return JSONResponse(
            {"error": "No model_id or gpu_memory provided"},
            status_code=400,
        )
    gpu_memory = int(params.gpu_memory)

    try:
        tgi = get_tgi_config(params.model_id, gpu_memory, params.num_gpus)
        if tgi is None:
            return JSONResponse(
                {
                    "error": f"Couldn't generate TGI config for {params.model_id}",
                },
                status_code=400,
            )
        return ConfigResponse(config=tgi)

    except Exception as e:
        logger.error(f"Error looking up {params.model_id}: {e}")
        return handle_exception(e)
